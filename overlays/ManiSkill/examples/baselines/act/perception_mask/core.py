"""State machine and controlled interventions. No simulator imports or labels."""
from dataclasses import dataclass, field
import time

import numpy as np

from .single_target_guard import SingleTargetGuard

SOURCE = 'vlm_sam2_cutie'


def validate_box(box, width, height, coordinate_space='pixels'):
    a = np.asarray(box, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError('VLM bbox must contain four finite XYXY values')
    if coordinate_space == 'normalized_1000':
        a *= np.array([width, height, width, height]) / 1000
    elif coordinate_space == 'normalized_1':
        a *= [width, height, width, height]
    elif coordinate_space != 'pixels':
        raise ValueError('Unknown bounding-box coordinate space')
    if not (0 <= a[0] < a[2] <= width and 0 <= a[1] < a[3] <= height):
        raise ValueError('VLM bbox is inverted, empty or outside the actual image')
    if min(a[2]-a[0], a[3]-a[1]) < 2:
        raise ValueError('VLM bbox is too small')
    return a.tolist()


def bbox_from_mask(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)] if len(x) else None


def options_from_dict(value):
    value = dict(value or {})
    allowed = {'loss_mode', 'intervention', 'min_pixels', 'min_quality', 'max_area_fraction',
               'lost_grace', 'max_reacquires', 'reacquire_cooldown', 'single_target', 'single_target_bbox'}
    if set(value) - allowed:
        raise ValueError('Unknown perception options')
    defaults = dict(loss_mode='observe', min_pixels=4, min_quality=0.55, max_area_fraction=0.75,
                    lost_grace=3, max_reacquires=2, reacquire_cooldown=15,
                    single_target=False, single_target_bbox=None)
    defaults.update(value)
    if defaults['loss_mode'] not in ('observe', 'pause', 'reacquire'):
        raise ValueError('Unknown lost-mask mode')
    if type(defaults['single_target']) is not bool:
        raise ValueError('single_target must be boolean')
    tight = defaults['single_target_bbox']
    if tight is not None:
        if not isinstance(tight, (list, tuple)) or len(tight) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in tight):
            raise ValueError('single_target_bbox must contain four finite pixel coordinates')
        if not (0 <= tight[0] < tight[2] and 0 <= tight[1] < tight[3]) or min(tight[2]-tight[0], tight[3]-tight[1]) < 2:
            raise ValueError('single_target_bbox must be a nonempty pixel box')
        if not defaults['single_target']:
            raise ValueError('single_target_bbox requires single_target=true')
        defaults['single_target_bbox'] = list(tight)
    for k, low, high in [('min_pixels',1,10000),('lost_grace',1,100),('max_reacquires',0,10),('reacquire_cooldown',1,1000)]:
        if type(defaults[k]) is not int or not low <= defaults[k] <= high:
            raise ValueError('Invalid '+k)
    for k in ('min_quality','max_area_fraction'):
        if not isinstance(defaults[k], (float,int)) or not 0 < defaults[k] <= 1:
            raise ValueError('Invalid '+k)
    intervention = dict(kind='none', start=50, duration=30, dx=20, dy=0)
    extra = defaults.get('intervention', {})
    if not isinstance(extra, dict) or set(extra)-set(intervention):
        raise ValueError('Unknown intervention settings')
    intervention.update(extra)
    if intervention['kind'] not in ('none','dropout','freeze','shift','occlusion'):
        raise ValueError('Unknown intervention kind')
    for k, low, high in [('start',0,100000),('duration',1,100000),('dx',-2048,2048),('dy',-2048,2048)]:
        if type(intervention[k]) is not int or not low <= intervention[k] <= high:
            raise ValueError('Invalid intervention '+k)
    defaults['intervention'] = intervention
    return defaults


def shift_mask(mask, dx, dy):
    out = np.zeros_like(mask)
    h,w = mask.shape
    sx0,sx1 = max(0,-dx), min(w,w-dx)
    sy0,sy1 = max(0,-dy), min(h,h-dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0+dy:sy1+dy,sx0+dx:sx1+dx] = mask[sy0:sy1,sx0:sx1]
    return out


@dataclass
class FrameResult:
    mask: np.ndarray
    metadata: dict


class TrackingSession:
    """Consumes only RGB, natural language and previous model outputs.

    Quality is a tracker heuristic, not correctness probability. GT IoU must be
    calculated outside this boundary and must never drive recovery decisions.
    """
    def __init__(self, locator, segmenter, tracker_factory, text, options=None, initial_localization=None):
        if not isinstance(text,str) or not text.strip() or len(text)>1000:
            raise ValueError('A visible natural-language target description is required')
        self.locator, self.segmenter, self.tracker_factory = locator, segmenter, tracker_factory
        self.text, self.options = text.strip(), options_from_dict(options)
        self.tracker = None
        self.generation = 0
        self.reacquires = 0
        self.last_acquire_frame = -100000
        self.lost_streak = 0
        self.last_frame = -1
        self.shape = None
        self.last_result = None
        self.last_good_mask = None
        self.frozen_mask = None
        self.occlusion_box = None
        self.vlm_bbox = None
        self.last_access = time.monotonic()
        self.initial_localization = initial_localization
        self.single_target_guard = (SingleTargetGuard(self.options['min_pixels'])
                                    if self.options['single_target'] else None)
        self._guard_info = dict(enabled=True, accepted=False, reason='uninitialized')

    def _active(self, frame):
        i = self.options['intervention']
        return i['kind']!='none' and i['start'] <= frame < i['start']+i['duration']

    def _image_for_perception(self, rgb, frame):
        if not self._active(frame) or self.options['intervention']['kind']!='occlusion':
            return rgb
        if self.occlusion_box is None:
            self.occlusion_box = bbox_from_mask(self.last_good_mask) if self.last_good_mask is not None else self.vlm_bbox
        box = self.occlusion_box
        if box is None:
            return rgb
        h,w = rgb.shape[:2]
        x0,y0,x1,y1 = [int(v) for v in box]
        image = rgb.copy()
        image[max(0,y0-6):min(h,y1+6),max(0,x0-6):min(w,x1+6)] = 0
        return image

    def _acquire(self, image, frame, events):
        self.last_acquire_frame = frame
        if self.single_target_guard is not None:
            self._guard_info = dict(enabled=True, accepted=False, reason='localization_not_available')
        upstream_seed = self.initial_localization is not None
        if upstream_seed:
            located,self.initial_localization=self.initial_localization,None
        else:
            located=self.locator.locate(image,self.text)
        events.append(dict(type='vlm_bbox',frame=frame,found=bool(located.get('found')),reason=located.get('reason',''),
                           origin=located.get('origin','perception_grounder'),request_id=located.get('request_id')))
        if not located.get('found'):
            return None, 0.0
        h,w = image.shape[:2]
        try:
            self.vlm_bbox = validate_box(located.get('bbox_xyxy'),w,h,located.get('coordinate_space','pixels'))
            # Task sessions may pad SAM2's prompt, while the guard must retain
            # the original tight VLM cue. A later grounder reacquisition gets
            # a new cue; never bind it to the old upstream box.
            tight = (self.options['single_target_bbox'] if upstream_seed
                     and self.options['single_target_bbox'] is not None else self.vlm_bbox)
            tight = validate_box(tight,w,h)
        except (ValueError,TypeError) as error:
            if self.single_target_guard is not None:
                self._guard_info = dict(enabled=True, accepted=False, reason='invalid_seed_bbox')
            events.append(dict(type='vlm_bbox_invalid',frame=frame,reason=str(error)))
            return None,0.0
        mask, quality = self.segmenter.segment(image,self.vlm_bbox)
        events.append(dict(type='sam2_segmented',frame=frame,
                           device=getattr(self.segmenter,'last_device',None),
                           fallback_reason=getattr(self.segmenter,'last_fallback_reason',None)))
        mask = np.asarray(mask,dtype=bool)
        if mask.shape != (h,w):
            raise ValueError('SAM2 mask has incorrect image dimensions')
        if not np.isfinite(quality):
            raise ValueError('SAM2 returned a nonfinite mask quality')
        if (int(mask.sum()) < self.options['min_pixels'] or float(quality) < self.options['min_quality']
                or mask.mean() > self.options['max_area_fraction']):
            if self.single_target_guard is not None:
                self._guard_info = dict(enabled=True, accepted=False, reason='sam2_seed_quality_rejected')
            events.append(dict(type='sam2_seed_rejected',frame=frame,pixels=int(mask.sum()),quality=float(quality)))
            return None,float(quality)
        if self.single_target_guard is not None:
            checked = self.single_target_guard.seed(mask,tight)
            self._guard_info = checked.metadata
            events.append(dict(type='single_target_seed_check',frame=frame,**checked.metadata))
            if not checked.accepted:
                return None,float(quality)
            mask = checked.mask
        new_tracker = self.tracker_factory()
        new_tracker.initialize(image,mask)
        if self.tracker is not None:
            self.tracker.close()
        self.tracker = new_tracker
        self.generation += 1
        self.last_good_mask = mask.copy()
        if self.single_target_guard is not None:
            self.single_target_guard.trust(mask,reset_anchor=True)
        events.append(dict(type='initialized' if self.generation==1 else 'reacquired',frame=frame,generation=self.generation,bbox_xyxy=self.vlm_bbox))
        return mask,float(quality)

    def process(self, rgb, frame, force_reacquire=False):
        started = time.monotonic()
        rgb = np.asarray(rgb)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3 or min(rgb.shape[:2])<16 or max(rgb.shape[:2])>2048:
            raise ValueError('Expected a uint8 RGB image with valid dimensions')
        if type(frame) is not int or frame < 0 or frame < self.last_frame:
            raise ValueError('Tracking frame index must be monotonically increasing')
        if self.shape is None:
            self.shape = rgb.shape[:2]
        elif self.shape != rgb.shape[:2]:
            raise ValueError('Image dimensions changed; start a new tracking session')
        self.last_access = started
        if frame == self.last_frame and not force_reacquire:
            return self.last_result
        events = []
        image = self._image_for_perception(rgb,frame)
        first = self.last_frame < 0
        due = self.options['loss_mode']=='reacquire' and self.lost_streak>=self.options['lost_grace'] and frame-self.last_acquire_frame>=self.options['reacquire_cooldown'] and self.reacquires<self.options['max_reacquires']
        acquiring = first or force_reacquire or due
        if force_reacquire or due:
            if self.reacquires >= self.options['max_reacquires']:
                events.append(dict(type='reacquire_budget_exhausted',frame=frame))
                acquiring = False
            else:
                self.reacquires += 1
        if acquiring:
            raw,quality = self._acquire(image,frame,events)
        elif self.tracker is not None:
            raw,quality = self.tracker.predict(image)
        else:
            raw,quality = None,0.0
        if raw is None:
            raw = np.zeros(self.shape,dtype=bool)
        raw = np.asarray(raw,dtype=bool)
        if raw.shape != self.shape or not np.isfinite(quality):
            raise ValueError('Tracker returned an invalid mask or quality')
        pixels = int(raw.sum())
        natural_lost = pixels<self.options['min_pixels'] or pixels/raw.size>self.options['max_area_fraction'] or float(quality)<self.options['min_quality']
        if self.single_target_guard is not None and not acquiring and natural_lost:
            self._guard_info = dict(enabled=True, accepted=False, reason='tracker_mask_quality_rejected')
        if self.single_target_guard is not None and not acquiring and not natural_lost:
            checked = self.single_target_guard.prediction(raw)
            self._guard_info = checked.metadata
            events.append(dict(type='single_target_tracking_check',frame=frame,**checked.metadata))
            if checked.accepted:
                raw = checked.mask
            else:
                natural_lost = True
        if natural_lost:
            if self.tracker is not None and not acquiring:
                self.tracker.reject()
        else:
            self.last_good_mask = raw.copy()
            if self.tracker is not None and not acquiring:
                self.tracker.accept()
                if self.single_target_guard is not None:
                    self.single_target_guard.trust(raw)
        effective = np.zeros_like(raw) if natural_lost else raw.copy()
        active = self._active(frame)
        kind = self.options['intervention']['kind']
        if active:
            if kind=='dropout':
                effective[:] = False
            elif kind=='freeze':
                if self.frozen_mask is None:
                    # Freeze the previous policy frame, not a future ground-truth mask.
                    self.frozen_mask = self.last_result.mask.copy() if self.last_result is not None else effective.copy()
                effective = self.frozen_mask.copy()
            elif kind=='shift':
                effective = shift_mask(effective,self.options['intervention']['dx'],self.options['intervention']['dy'])
        lost = natural_lost or not effective.any()
        self.lost_streak = self.lost_streak+1 if lost else 0
        previous_state = self.last_result.metadata['state'] if self.last_result is not None else 'initializing'
        state = 'lost' if lost else 'tracking'
        if state != previous_state:
            events.append(dict(type='state_change',frame=frame,previous=previous_state,state=state))
        metadata = dict(source=SOURCE,state=state,frame_index=frame,generation=self.generation,
                        quality=float(quality),quality_kind='sam2_predicted_iou' if acquiring else 'cutie_foreground_probability_heuristic',
                        bbox_xyxy=bbox_from_mask(effective),vlm_bbox_xyxy=self.vlm_bbox,
                        tracked_pixels=pixels,mask_pixels=int(effective.sum()),natural_lost=bool(natural_lost),
                        lost_streak=self.lost_streak,reacquires=self.reacquires,
                        intervention=kind if active else 'none',intervention_active=active,
                        latency_ms=round((time.monotonic()-started)*1000,3),events=events,
                        segmentation_used_as_input=False)
        if self.single_target_guard is not None:
            metadata['single_target_guard'] = dict(self._guard_info)
        self.last_frame = frame
        self.last_result = FrameResult(effective,metadata)
        return self.last_result

    def close(self):
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
