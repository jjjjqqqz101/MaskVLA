"""Qwen calls see pixels and task history, without simulator annotations/state."""
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from PIL import Image,ImageDraw,ImageFont

from agentic_act.qwen import QwenClient
from .contracts import validate_plan,validate_target,validate_judgement,validate_completion,validate_mask_check


class TaskVLM:
    def __init__(self,instruction,output,config=None,max_steps=4):
        self.instruction=instruction
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.client=QwenClient(config=config,output=None,timeout=15)
        self.max_steps=max_steps
        self.lock=threading.RLock();self.calls=0

    @staticmethod
    def _image(rgb):
        # Upscaling is a display transform only. Coordinates use a normalized
        # 1000-grid and are explicitly mapped back to the original RGB.
        image=Image.fromarray(rgb).convert('RGB')
        scale=min(3,672/max(image.size))
        return image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.BILINEAR)

    @staticmethod
    def _overview(snapshot):
        return snapshot.get('overview_rgb',snapshot['base_rgb'])

    @classmethod
    def _ground_image(cls,rgb,longest_edge=None):
        # A coordinate ruler improves small-object localization without adding
        # object identities, simulator geometry or truth masks to the image.
        if longest_edge is None:
            image=cls._image(rgb).copy()
        else:
            image=Image.fromarray(rgb).convert('RGB')
            scale=longest_edge/max(image.size)
            image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),
                               Image.Resampling.BILINEAR)
        draw=ImageDraw.Draw(image)
        font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        font=ImageFont.truetype(str(font_path),15) if font_path.is_file() else ImageFont.load_default()
        for coordinate in range(0,1001,100):
            x=min(image.width-1,round(coordinate*image.width/1000))
            y=min(image.height-1,round(coordinate*image.height/1000))
            draw.line((x,0,x,image.height-1),fill=(100,220,245),width=1)
            draw.line((0,y,image.width-1,y),fill=(100,220,245),width=1)
            draw.text((min(x+3,image.width-42),3),str(coordinate),font=font,fill='white',stroke_width=1,stroke_fill='black')
            draw.text((3,min(y+3,image.height-22)),str(coordinate),font=font,fill='white',stroke_width=1,stroke_fill='black')
        return image

    def _request(self,purpose,system,prompt,snapshots,images):
        with self.lock:
            self.calls+=1;request_id=f'task_vlm_{self.calls:06d}'
        started=time.monotonic()
        record=dict(request_id=request_id,purpose=purpose,model=self.client.model,started_at=started,
            frame_ids=[s['frame_id'] for s in snapshots],
            image_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in images],
            image_sizes=[list(im.size) for im in images],
            captured_at=[s['captured_at'] for s in snapshots])
        try:
            response=self.client.request(purpose,system,json.dumps(prompt,ensure_ascii=False),images)
            record['response']=response
        except Exception as exc:
            record['error']=self.client.redact(str(exc));raise
        finally:
            record.update(completed_at=time.monotonic(),latency_ms=(time.monotonic()-started)*1000)
            with self.lock:
                with (self.output/'vlm_calls.jsonl').open('a') as stream:
                    stream.write(self.client.redact(json.dumps(record,ensure_ascii=False,allow_nan=False))+'\n')
        return response,request_id

    def plan(self,snapshot):
        data,_=self._request('task_plan',
            'You are a robot task planner. Inspect the unannotated RGB image and decompose the user task. '
            'Available skill: learned grasp/lift followed by calibrated motion-planner transport, release and retreat. '
            'There is one visible green collection area, which is the default destination for cleanup. '
            'Choose matching objects yourself from visible shape/appearance; do not request simulator IDs or user checkboxes. '
            'Do not move unrelated objects. Distinguish spheres/balls from cubes, cups and packaging. '
            'Inspect all supplied views jointly, including small and partly occluded objects; size alone does not exclude a target. '
            'Image order: overview, base camera, wrist camera. The same object may appear in several views; count it once. '
            f'The area holds at most {self.max_steps} objects. If a task needs unavailable skills or destinations, set needs_clarification=true. '
            'Return JSON {"summary":"简要计划","selection_rule":"目标选择规则",'
            '"needs_clarification":false,"steps":[{"skill":"pick_place","description":"visible target description",'
            '"reason":"why this object matches"}]}. '
            'Use Chinese explanations. Steps are proposed visual targets, not evidence of success. '
            'Do not return bounding boxes in this plan; only the currently executing subtask is grounded later.',
            dict(instruction=self.instruction,default_destination='visible green collection area'),
            [snapshot],[self._image(self._overview(snapshot)),self._image(snapshot['base_rgb']),self._image(snapshot['wrist_rgb'])])
        return validate_plan(data,self.max_steps)

    def target(self,snapshot,plan,completed,pending_hint=None):
        h,w=snapshot['base_rgb'].shape[:2]
        data,request_id=self._request('subtask_bbox',
            'You perform precise visual grounding. Select ONE currently visible remaining object that matches the requested task. '
            'Return exactly one tight bbox around exactly one physical object. Never group multiple similar objects in one box, '
            'and never return a list of targets or boxes. Other matching objects wait for later subtasks. '
            'Objects already entirely inside the green collection area are collected and must not be picked again. '
            'Use raw visible appearance and spatial relations, not simulator instance numbers. '
            'Prior completion records are fallible visual claims; correct them if current pixels contradict them. '
            'Follow any requested order and prioritize pending_hint. On a retry re-localize the same object instead of switching targets. '
            'The FIRST image is the base camera; bounding boxes MUST refer to this FIRST image. '
            'Only the base image is supplied for this grounding call. Check the exact silhouette and enclose the whole object, not nearby table pixels. '
            'The thin numbered grid is ONLY a normalized coordinate ruler, not object annotations. Use it to read the actual visible object boundaries. '
            'If pending_hint includes a failed box, its coordinate space is explicitly labeled; the normalized_1000 variant can be compared directly to the ruler. '
            'Locate the target with a tight bounding box on a normalized 0..1000 coordinate grid: '
            'top-left=(0,0), bottom-right=(1000,1000), x rightward, y downward. '
            'If no matching object remains outside the green area, return decision=complete. '
            'If occluded or ambiguous, return decision=uncertain. '
            'Return JSON {"decision":"pick|complete|uncertain","description":"visible target",'
            '"bbox_xyxy":[x0,y0,x1,y1],"coordinate_space":"normalized_1000","reason":"Chinese visual evidence"}.',
            dict(instruction=self.instruction,plan=plan,previous_visual_completions=completed,pending_hint=pending_hint),
            [snapshot],[self._ground_image(snapshot['base_rgb'])])
        result=validate_target(data,w,h)
        result.update(request_id=request_id,source_frame_id=snapshot['frame_id'])
        if result['decision']!='pick':return result

        # The first call selects the object. A second view enlarges its local
        # context so small silhouettes are grounded from pixels, without an
        # object-specific offset, simulator state or a coarse-box fallback.
        coarse=result['bbox_xyxy']
        margin=max(24.,2*max(coarse[2]-coarse[0],coarse[3]-coarse[1]))
        x0=max(0,math.floor(coarse[0]-margin));y0=max(0,math.floor(coarse[1]-margin))
        x1=min(w,math.ceil(coarse[2]+margin));y1=min(h,math.ceil(coarse[3]+margin))
        crop=snapshot['base_rgb'][y0:y1,x0:x1]
        intended=pending_hint.get('description') if isinstance(pending_hint,dict) else None
        intended=intended.strip() if isinstance(intended,str) and intended.strip() else None
        identity=intended or result['description']
        refined,refined_id=self._request('subtask_bbox_refine',
            'Precisely locate the SAME requested object in this ONE enlarged RGB crop. '
            'The task planner already selected its identity; do not switch to a different object. '
            'The object must match BOTH coarse_target_description and planned_or_retry_target_description '
            '(when supplied), and must respect the current plan/order. '
            'If the coarse call selected a different category or identity from the planned/retry target, '
            'return found=false even if that other object is visible. Judge identity from visual semantics, not object IDs. '
            'Spatial words may describe the original position before movement; preserve the intended object identity. '
            'The thin numbered grid is only a normalized coordinate ruler for this crop. '
            'Read the actual visible silhouette and return a tight box enclosing the entire requested object, not table/background. '
            'Coordinates refer ONLY to this crop: top-left=(0,0), bottom-right=(1000,1000), '
            'x rightward, y downward. Ignore coordinates from the earlier full scene. '
            'If this same object is absent, occluded, or ambiguous, return found=false instead of choosing another object. '
            'Return JSON {"found":true,"bbox_xyxy":[x0,y0,x1,y1],'
            '"coordinate_space":"normalized_1000","reason":"Chinese visible boundary evidence"}. '
            'For found=false use bbox_xyxy=null and explain the visible uncertainty.',
            dict(target_description=identity,coarse_target_description=result['description'],
                 planned_or_retry_target_description=intended,plan=plan,pending_hint=pending_hint,
                 instruction=self.instruction,
                 coordinate_space='normalized_1000',image_scope='local RGB crop only'),
            [snapshot],[self._ground_image(crop,longest_edge=672)])
        metadata=dict(description=identity,coarse_description=result['description'],coarse_bbox_xyxy=coarse,
            crop_xyxy=[x0,y0,x1,y1],coarse_request_id=request_id,
            coarse_reason=result['reason'],request_id=refined_id,refined_request_id=refined_id,
            source_frame_id=snapshot['frame_id'])
        try:
            if not isinstance(refined,dict) or type(refined.get('found')) is not bool:
                raise ValueError('Refined localization must explicitly return found=true/false')
            if not refined['found']:
                uncertain=validate_target(dict(decision='uncertain',reason=refined.get('reason')),x1-x0,y1-y0)
                return dict(metadata,**uncertain)
            if refined.get('coordinate_space','normalized_1000')!='normalized_1000':
                raise ValueError('Refined box must use the crop normalized_1000 coordinate ruler')
            local=validate_target(dict(decision='pick',description=identity,
                bbox_xyxy=refined.get('bbox_xyxy'),coordinate_space=refined.get('coordinate_space','normalized_1000'),
                reason=refined.get('reason')),x1-x0,y1-y0)
        except (ValueError,TypeError) as error:
            return dict(metadata,decision='uncertain',reason='Fine localization unusable: '+str(error))
        local_box=local['bbox_xyxy']
        final_box=[local_box[0]+x0,local_box[1]+y0,local_box[2]+x0,local_box[3]+y0]
        # Revalidate in the native base-camera image before handing off to SAM2.
        final=validate_target(dict(decision='pick',description=identity,
            reason=local['reason'],bbox_xyxy=final_box,coordinate_space='pixels'),w,h)
        return dict(metadata,**final)

    def check_mask(self,snapshot):
        rgb=snapshot['base_rgb'];mask=snapshot.get('mask')
        if mask is None or not mask.any():return dict(valid=False,reason='SAM2 returned no visible target pixels')
        masked=np.where(mask[...,None],rgb,0).astype(np.uint8)
        yy,xx=np.nonzero(mask)
        crop=masked[max(0,yy.min()-2):min(rgb.shape[0],yy.max()+3),max(0,xx.min()-2):min(rgb.shape[1],xx.max()+3)]
        crop_image=Image.fromarray(crop)
        crop_scale=256/max(crop_image.size)
        crop_image=crop_image.resize((max(1,round(crop_image.width*crop_scale)),max(1,round(crop_image.height*crop_scale))),Image.Resampling.BILINEAR)
        data,_=self._request('mask_target_confirmation',
            'Before any robot action, verify whether SAM2 segmented the requested object. '
            'Image order: full raw base RGB, segmented crop enlarged, full masked RGB. '
            'Return valid=true only if the segmented pixels visibly show the requested object silhouette/appearance. '
            'A patch of wooden table, background, another object, empty or ambiguous segmentation must be valid=false. '
            'The segmentation must contain exactly ONE requested physical object. If a second object is also included, '
            'even another object of the same category, return valid=false. '
            'Do not infer correct segmentation just because the object is nearby in the raw image. '
            'Return JSON {"valid":false,"reason":"简短中文视觉证据"}.',
            dict(target=snapshot.get('target_text'),vlm_bbox_xyxy=snapshot.get('mask_metadata',{}).get('vlm_bbox_xyxy')),
            [snapshot],[self._image(rgb),crop_image,self._image(masked)])
        return validate_mask_check(data)

    def judge(self,snapshot,previous=None):
        images=[];frames=[];order=[]
        phase_rules=(
            'CURRENT PHASE IS PLACEMENT VERIFICATION. Judge ONLY the current_target identity, regardless of other unfinished objects. '
            'An empty gripper after release is EXPECTED. If the current target is inside green and released, '
            'assessment MUST be placed, even if other task objects remain outside. '
            'Do not use not_grasped or grasped_lifted in this phase. Use placed/failed/uncertain only. '
            'Spatial words in current_target describe its original identity; the object may now have moved. '
            if snapshot.get('phase')=='verify' else
            'Judge only the current_target identity. Other remaining objects do not change whether this target was grasped. ')
        if snapshot.get('phase')=='transport':
            phase_rules+=(
                'Read motion_phase as commanded intent, not proof of success. Before release the target should be held. '
                'During release, retreat_raise, retreat_joint_home, or settle, an empty gripper is EXPECTED after intentional release. '
                'If the target is visibly inside green and released, report placed/continue, not grasp failure. '
                'After release, judge whether the target is placed correctly; retry requires evidence such as outside the region or wrong object. ')
        if previous is not None:
            images.append(self._image(self._overview(previous)));frames.append(previous);order.append('earlier overview view')
        images.append(self._image(self._overview(snapshot)));order.append('current overview view')
        images.append(self._image(snapshot['base_rgb']));frames.append(snapshot);order.append('current base view')
        if snapshot.get('wrist_rgb') is not None:
            images.append(self._image(snapshot['wrist_rgb']));order.append('current wrist view')
        mask=snapshot.get('mask')
        if mask is not None and snapshot.get('target_text'):
            masked=np.where(mask[...,None],snapshot['base_rgb'],0).astype(np.uint8)
            images.append(self._image(masked));order.append('current single-target masked base RGB')
        data,_=self._request('visual_supervision_1hz',
            phase_rules+'You visually supervise a real execution in a simulator. Judge only the images, including motion between views. '
            'The command phase is intention, not proof that anything succeeded. Do not invent a grasp from proximity. '
            'grasped_lifted requires the requested object visibly held between fingers AND lifted off the table; '
            'placed requires that object visibly inside the green area, released on the table, with gripper separated. '
            'When the gripper hides the object or evidence is ambiguous return uncertain/continue. '
            'A ball outside the green area is still unfinished. An object slipping or a wrong object held can justify retry. '
            'During transport, intended release inside the collection area is normal. '
            'During grasp, decision=next/complete cannot finish a task. '
            'The masked base image, when supplied, should show only the current single physical target. '
            'If it clearly includes a second distinct object or switches to a wrong object during grasp, '
            'report failed/retry and explain target-mask contamination. '
            'An empty or partly occluded mask alone is not proof of grasp failure: judge the raw scene and motion. '
            'Return JSON {"assessment":"not_grasped|grasped_lifted|placed|failed|uncertain",'
            '"decision":"continue|retry|abort|next|complete","confidence":0.0,"reason":"Chinese visible evidence"}. '
            'Use continue for normal ongoing progress, retry for clear failure, abort for unrecoverable problems. '
            'Your confidence is a visual opinion, not a physical contact sensor. Keep reason under 50 Chinese characters.',
            dict(instruction=self.instruction,current_target=snapshot.get('target_text',''),
                 phase=snapshot.get('phase'),motion_phase=snapshot.get('motion_phase'),
                 frame_id=snapshot['frame_id'],image_order=order),frames,images)
        return validate_judgement(data)

    def finish(self,snapshot,completed):
        data,_=self._request('task_completion',
            'Judge the whole requested cleanup task from the current unannotated RGB image. '
            'Previous step successes are fallible model claims. Inspect all visible matching objects. '
            'Complete requires all requested visible targets inside the green collection area and released; '
            'a matching object still outside, held by the robot, or hidden with unresolved identity makes complete=false. '
            'Do not confuse cubes/cups with balls. Return JSON {"complete":true,"reason":"Chinese visual evidence",'
            '"remaining_description":"visible remaining targets, or empty"}.',
            dict(instruction=self.instruction,previous_visual_completions=completed),
            [snapshot],[self._image(self._overview(snapshot)),self._image(snapshot['base_rgb'])])
        return validate_completion(data)
