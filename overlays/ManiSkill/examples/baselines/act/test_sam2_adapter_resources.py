"""SAM2 resource routing contracts; do not allocate GPU memory or load weights."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch

from perception_mask.adapters import SAM2Segmenter


class SAM2ResourceContracts(unittest.TestCase):
    def make(self,device='cuda'):
        adapter=SAM2Segmenter(checkpoint=Path(__file__),device=device)
        adapter.predictor=SimpleNamespace(reset_predictor=Mock())
        return adapter

    def test_less_than_two_gib_routes_cpu(self):
        adapter=self.make()
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.mem_get_info',return_value=(2*1024**3-1,24*1024**3)):
            self.assertEqual(adapter._execution_device(torch),'cpu')
        self.assertEqual(adapter.last_fallback_reason,'low_free_cuda_memory')

    def test_exact_two_gib_keeps_requested_cuda(self):
        adapter=self.make('cuda:1')
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.mem_get_info',return_value=(2*1024**3,24*1024**3)) as memory:
            self.assertEqual(adapter._execution_device(torch),'cuda:1')
        self.assertEqual(memory.call_args.args,(torch.device('cuda:1'),))
        self.assertIsNone(adapter.last_fallback_reason)

    def test_explicit_cpu_does_not_touch_cuda(self):
        adapter=self.make('cpu')
        with patch('torch.cuda.is_available') as available,patch('torch.cuda.mem_get_info') as memory:
            self.assertEqual(adapter._execution_device(torch),'cpu')
        available.assert_not_called();memory.assert_not_called()

    def test_unavailable_cuda_routes_cpu(self):
        adapter=self.make()
        with patch('torch.cuda.is_available',return_value=False):self.assertEqual(adapter._execution_device(torch),'cpu')
        self.assertEqual(adapter.last_fallback_reason,'cuda_unavailable')

    def test_cuda_oom_retries_same_image_and_box_on_cpu(self):
        adapter=self.make();adapter._execution_device=Mock(return_value='cuda')
        adapter._offload_predictor=Mock()
        image=np.zeros((4,4,3),np.uint8);box=[0,0,3,3];mask=np.ones((4,4),bool)
        adapter._infer_once=Mock(side_effect=[torch.cuda.OutOfMemoryError('CUDA OOM'),(mask,.8)])
        actual,score=adapter.segment(image,box)
        self.assertIs(actual,mask);self.assertEqual(score,.8)
        self.assertEqual([call.args[2] for call in adapter._infer_once.call_args_list],['cuda','cpu'])
        for call in adapter._infer_once.call_args_list:
            self.assertIs(call.args[0],image);self.assertIs(call.args[1],box)
        adapter.predictor.reset_predictor.assert_called_once()
        adapter._offload_predictor.assert_called_once_with(torch,empty_cuda_cache=True)
        self.assertEqual(adapter.last_fallback_reason,'cuda_oom')

    def test_cpu_oom_is_not_retried(self):
        adapter=self.make('cpu');adapter._infer_once=Mock(side_effect=torch.cuda.OutOfMemoryError('CPU allocation failed'))
        adapter._offload_predictor=Mock()
        with self.assertRaises(torch.cuda.OutOfMemoryError):adapter.segment(np.zeros((4,4,3),np.uint8),[0,0,3,3])
        self.assertEqual(adapter._infer_once.call_count,1);adapter._offload_predictor.assert_not_called()

    def test_other_errors_are_not_disguised_as_resource_fallback(self):
        adapter=self.make();adapter._execution_device=Mock(return_value='cuda')
        adapter._infer_once=Mock(side_effect=RuntimeError('Invalid image / bad weights'))
        adapter._offload_predictor=Mock()
        with self.assertRaisesRegex(RuntimeError,'Invalid image'):adapter.segment(np.zeros((4,4,3),np.uint8),[0,0,3,3])
        self.assertEqual(adapter._infer_once.call_count,1);adapter._offload_predictor.assert_not_called()

    def test_offload_clears_sam_plain_caches(self):
        adapter=self.make()
        module=SimpleNamespace(cache={'old':torch.ones(1)},_cache={'old':torch.ones(1)},freqs_cis=torch.ones(1))
        model=Mock();model.modules.return_value=[module]
        adapter.predictor.model=model
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.empty_cache') as empty:
            adapter._offload_predictor(torch,empty_cuda_cache=True)
        model.to.assert_called_once_with('cpu');self.assertEqual(module.cache,{});self.assertEqual(module._cache,{})
        self.assertEqual(module.freqs_cis.device.type,'cpu');empty.assert_called_once()


if __name__=='__main__':unittest.main()
