"""Pure initial-layout checks; no simulator, model or rendering startup."""
import math
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from task_agent.scene_objects import CLASSIC_OBJECT_IDS,get_object_spec,resolve_scene_spec
from task_agent.scene_env import pack_layout,WORKSPACE_MIN,WORKSPACE_MAX,CENTER_MIN,CENTER_MAX


class SelectableLayoutContracts(unittest.TestCase):
    def assert_geometry(self,specs,layout):
        for spec,item in zip(specs,layout):
            low=np.asarray(item['aabb_min']);high=np.asarray(item['aabb_max']);center=(low+high)/2
            self.assertTrue((low>=WORKSPACE_MIN-1e-10).all());self.assertTrue((high<=WORKSPACE_MAX+1e-10).all())
            self.assertTrue((center>=CENTER_MIN).all());self.assertTrue((center<=CENTER_MAX+1e-10).all())
            self.assertGreaterEqual(float(low[0]),-.12-1e-10)
            if spec['kind']=='ycb':self.assertEqual(item['yaw'],0.)
            else:self.assertTrue(0<=item['yaw']<=2*math.pi)
        for i,left in enumerate(layout):
            for right in layout[i+1:]:
                a,b=np.asarray(left['aabb_min']),np.asarray(left['aabb_max'])
                c,d=np.asarray(right['aabb_min']),np.asarray(right['aabb_max'])
                self.assertTrue(b[0]+.01<=c[0]+1e-10 or d[0]+.01<=a[0]+1e-10
                    or b[1]+.01<=c[1]+1e-10 or d[1]+.01<=a[1]+1e-10)

    def test_custom_eight_packing_reproducible_bounded_and_nonoverlapping(self):
        specs=[get_object_spec(name) for name in CLASSIC_OBJECT_IDS]
        for seed in range(20):
            with self.subTest(seed=seed):
                first=pack_layout(specs,seed=seed)
                self.assertEqual(first,pack_layout(specs,seed=seed))
                self.assert_geometry(specs,first)

    def test_large_catalog_footprints_fit_only_inside_reachable_bounds(self):
        names=['024_bowl','011_banana','003_cracker_box','036_wood_block','025_mug']
        specs=[get_object_spec(name) for name in names]
        for seed in (1,17,4090):self.assert_geometry(specs,pack_layout(specs,seed=seed))

    def test_impossible_static_size_fails_without_out_of_bounds_fallback(self):
        impossible=dict(kind='ycb',bbox_min=[-.25,-.25,-.02],bbox_max=[.25,.25,.02])
        with self.assertRaisesRegex(ValueError,'无法.*无重叠摆放'):
            pack_layout([impossible],seed=17,layout_attempts=2,candidate_attempts=2)
        for specs in ([],[get_object_spec('013_apple')]*9):
            with self.assertRaises(ValueError):pack_layout(specs,seed=17)

    def test_backend_custom_constructor_selects_independent_env_and_checks_actual_ids(self):
        from task_agent.backend import RobotBackend
        ids=['025_mug','013_apple']
        canonical=resolve_scene_spec({'mode':'manual','object_ids':ids})
        observation={'agent':{'qpos':np.zeros((1,9),np.float32)}}
        for match in (True,False):
            with self.subTest(actual_match=match),TemporaryDirectory() as directory:
                actors=[SimpleNamespace(name=name) for name in (ids if match else ids[:1])]
                raw=SimpleNamespace(unwrapped=SimpleNamespace(cubes=actors,agent=SimpleNamespace(tcp=SimpleNamespace(
                    pose=SimpleNamespace(sp=object())))),reset=Mock(return_value=(observation,{})),close=Mock())
                raw.unwrapped.backend=SimpleNamespace(render_backend='sapien_cpu')
                with patch.dict('os.environ', {'MASK_RENDER_BACKEND':'cpu', 'MASK_SAPIEN_RENDERER':'lavapipe'}), \
                     patch('gymnasium.make',return_value=raw) as make, \
                     patch('task_agent.scene_env.register_selectable_scene') as register, \
                     patch('imageio.v2.get_writer',return_value=SimpleNamespace(close=Mock())), \
                     patch.object(RobotBackend,'_prepare_policy') as load, \
                     patch.object(RobotBackend,'_publish'),patch.object(RobotBackend,'_step'), \
                     patch.object(RobotBackend,'_save_image'):
                    if match:
                        backend=RobotBackend(seed=17,output=directory,scene_spec=canonical,defer_policy_loading=True)
                        self.assertEqual(backend.selected_object_ids,ids);self.assertEqual(backend.reset_count,1)
                        config=json.loads((Path(directory)/'backend_config.json').read_text())
                        self.assertEqual(config['render_backend'],'sapien_cpu');self.assertEqual(config['renderer'],'lavapipe')
                        backend.close()
                    else:
                        with self.assertRaisesRegex(RuntimeError,'do not match'):
                            RobotBackend(seed=17,output=directory,scene_spec=canonical,defer_policy_loading=True)
                    self.assertEqual(make.call_args.args[0],'TaskCollectionSelectable-v1')
                    self.assertEqual(make.call_args.kwargs['scene_objects'],canonical['objects'])
                    self.assertEqual(make.call_args.kwargs['render_backend'],'cpu')
                    register.assert_called_once();load.assert_not_called()


if __name__=='__main__':unittest.main()
