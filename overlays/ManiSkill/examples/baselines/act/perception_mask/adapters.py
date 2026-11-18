"""Real Qwen grounding, official SAM2 image predictor, official Cutie tracker."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from deployment_paths import (policy_root, sam2_checkpoint, cutie_checkpoint,
                              sam2_repo, cutie_repo, project_path)
from .core import validate_box

BASE = policy_root()


class QwenBoxLocator:
    def __init__(self, config=None, output=None):
        from agentic_act.qwen import QwenClient
        self.client = QwenClient(config=config, output=output, timeout=40)

    def locate(self, image, description):
        h,w = image.shape[:2]
        result = self.client.request('target_bbox',
            'Locate exactly one requested grasp object in this unannotated RGB image. '
            'Use visible appearance and spatial relations only. The image contains no trusted instance numbers. '
            'If absent, fully occluded or ambiguous, return found=false. Never invent hidden object locations. '
            'Return JSON {"found":true,"bbox_xyxy":[x0,y0,x1,y1],"coordinate_space":"pixels",'
            '"confidence":0.9,"reason":"brief visible evidence"}. '
            'Coordinates must be actual image pixels with top-left origin, x rightward and y downward. '
            'For found=false return bbox_xyxy=null. No markdown.',
            f'Image width={w}, height={h}. Target description: {description}',
            [Image.fromarray(image)])
        if type(result.get('found')) is not bool:
            return dict(found=False,reason='VLM returned an invalid found field',bbox_xyxy=None)
        if result['found']:
            try:
                result['bbox_xyxy'] = validate_box(result.get('bbox_xyxy'), w, h,
                                                  result.get('coordinate_space','pixels'))
            except (ValueError,TypeError) as error:
                # A bad model localization is an observed perception failure,
                # not a valid prompt and not a reason to substitute oracle data.
                return dict(found=False,reason='Invalid VLM localization: '+str(error),bbox_xyxy=None)
            result['coordinate_space'] = 'pixels'
        return result


class SAM2Segmenter:
    MIN_CUDA_FREE_BYTES = 2 * 1024**3

    def __init__(self, checkpoint=None, model_config='configs/sam2.1/sam2.1_hiera_s.yaml',
                 repo=None, device='cuda', offload=True):
        self.checkpoint = project_path(checkpoint or sam2_checkpoint(), 'SAM2 checkpoint')
        self.model_config = model_config
        self.repo = project_path(repo or sam2_repo(), 'SAM2 repo')
        self.device, self.offload = device,offload
        self.predictor = None
        self.last_device = None
        self.last_fallback_reason = None

    def _execution_device(self, torch):
        preferred = str(self.device)
        self.last_fallback_reason = None
        if not preferred.startswith('cuda'):
            return preferred
        if not torch.cuda.is_available():
            self.last_fallback_reason = 'cuda_unavailable'
            return 'cpu'
        free,_ = torch.cuda.mem_get_info(torch.device(preferred))
        if free < self.MIN_CUDA_FREE_BYTES:
            self.last_fallback_reason = 'low_free_cuda_memory'
            return 'cpu'
        return preferred

    def _offload_predictor(self, torch, empty_cuda_cache=False):
        model = self.predictor.model
        model.to('cpu')
        # Position encoders also retain device tensors outside registered
        # buffers. Release only SAM2 state; Cutie's GPU network stays in place.
        for module in model.modules():
            for name in ('cache','_cache'):
                cache = getattr(module,name,None)
                if isinstance(cache,dict):cache.clear()
            frequencies = getattr(module,'freqs_cis',None)
            if isinstance(frequencies,torch.Tensor):module.freqs_cis = frequencies.cpu()
        if empty_cuda_cache and torch.cuda.is_available():torch.cuda.empty_cache()

    def _infer_once(self, image, box, device, torch):
        self.last_device = device
        model = self.predictor.model
        try:
            model.to(device).eval()
            for module in model.modules():
                frequencies = getattr(module,'freqs_cis',None)
                if isinstance(frequencies,torch.Tensor):module.freqs_cis = frequencies.to(device)
            device_type = torch.device(device).type
            with torch.inference_mode(), torch.autocast(device_type,dtype=torch.bfloat16,enabled=device_type=='cuda'):
                self.predictor.set_image(image)
                masks,scores,_ = self.predictor.predict(box=np.asarray(box,dtype=np.float32),multimask_output=False)
                index = int(np.argmax(scores))
                return np.asarray(masks[index],dtype=bool).copy(),float(scores[index])
        finally:
            self.predictor.reset_predictor()
            if self.offload:
                self._offload_predictor(torch,empty_cuda_cache=device.startswith('cuda'))

    def segment(self, image, box):
        import torch
        if not self.checkpoint.is_file():
            raise RuntimeError('SAM2 checkpoint not found: '+str(self.checkpoint))
        if self.predictor is None:
            sys.path.insert(0,str(self.repo))
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(self.model_config,str(self.checkpoint),device='cpu',
                               apply_postprocessing=False)
            self.predictor = SAM2ImagePredictor(model)
        device = self._execution_device(torch)
        try:
            return self._infer_once(image,box,device,torch)
        except torch.cuda.OutOfMemoryError:
            if not device.startswith('cuda'):raise
            self.predictor.reset_predictor()
            self._offload_predictor(torch,empty_cuda_cache=True)
            self.last_fallback_reason = 'cuda_oom'
        # Leave the OOM exception/traceback scope before trying the same model,
        # image and box on CPU. Other errors propagate without fallback.
        return self._infer_once(image,box,'cpu',torch)


class CutieFactory:
    def __init__(self, checkpoint=None, repo=None, device='cuda'):
        self.checkpoint = project_path(checkpoint or cutie_checkpoint(), 'Cutie checkpoint')
        self.repo = project_path(repo or cutie_repo(), 'Cutie repo')
        self.device = device
        self.network = None
        self.config = None

    def __call__(self):
        if self.network is None:
            import torch
            sys.path.insert(0,str(self.repo))
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from cutie.model.cutie import CUTIE
            from cutie.inference.utils.args_utils import get_dataset_cfg
            from cutie.model.utils import resnet
            global_hydra = GlobalHydra.instance()
            previous_hydra = global_hydra.hydra
            global_hydra.clear()
            try:
                with initialize_config_dir(version_base='1.3', config_dir=str(self.repo/'cutie/config')):
                    config = compose(config_name='eval_config',overrides=['dataset=generic'])
            finally:
                global_hydra.hydra = previous_hydra
            get_dataset_cfg(config)
            self.config = config
            originals = {name:getattr(resnet,name) for name in ('resnet18','resnet50')}
            def without_pretrained(original):
                def build(*args,**kwargs):
                    kwargs['pretrained'] = False
                    return original(*args,**kwargs)
                return build
            try:
                for name,original in originals.items():
                    setattr(resnet,name,without_pretrained(original))
                network = CUTIE(config)
            finally:
                for name,original in originals.items():setattr(resnet,name,original)
            weights = torch.load(self.checkpoint,map_location='cpu',weights_only=True)
            network.load_state_dict(weights,strict=True)
            self.network = network.to(self.device).eval()
        return CutieTracker(self.network,self.config,self.device)


class CutieTracker:
    def __init__(self, network, config, device):
        from cutie.inference.inference_core import InferenceCore
        self.core = InferenceCore(network,cfg=config)
        self.core.max_internal_size = 480
        self.device = device
        self.trusted = None
        self.candidate = None
        self.trusted_sensory = {}

    def _tensor(self, image):
        import torch
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).to(self.device).float()/255

    def initialize(self, image, mask):
        import torch
        with torch.inference_mode(),torch.autocast(self.device,dtype=torch.float16,enabled=self.device.startswith('cuda')):
            self.core.step(self._tensor(image),torch.from_numpy(mask.astype(np.int64)).to(self.device),
                           objects=[1],force_permanent=True)
            self.trusted = self.core.last_mask.detach().clone()
            self.trusted_sensory = {k:v.detach().clone() for k,v in self.core.memory.sensory.items()}

    def predict(self, image):
        import torch
        with torch.inference_mode(),torch.autocast(self.device,dtype=torch.float16,enabled=self.device.startswith('cuda')):
            probabilities = self.core.step(self._tensor(image))
            labels = self.core.output_prob_to_mask(probabilities)
            mask = labels == 1
            quality = float(probabilities[1][mask].float().mean().item()) if mask.any() else 0.0
            self.candidate = self.core.last_mask.detach().clone()
            return mask.cpu().numpy().astype(bool), quality

    def accept(self):
        self.trusted = self.candidate
        self.trusted_sensory = {k:v.detach().clone() for k,v in self.core.memory.sensory.items()}

    def reject(self):
        # Keep the trusted SAM2 seed; discard working/sensory memory that might
        # already contain this low-quality frame. Never insert an empty GT mask.
        self.core.clear_non_permanent_memory()
        self.core.memory.sensory = {k:v.detach().clone() for k,v in self.trusted_sensory.items()}
        if self.trusted is not None:
            self.core.last_mask = self.trusted.detach().clone()

    def close(self):
        self.core = None
        self.trusted = self.candidate = None
        self.trusted_sensory = {}
