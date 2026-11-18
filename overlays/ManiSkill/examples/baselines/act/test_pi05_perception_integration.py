"""CPU contract tests for π0.5 perception experiments; no policy or service required.

Run in the existing ManiSkill Python environment:
    python -m unittest -v test_pi05_perception_integration
"""
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import pi05_target_web as web
import pi05_inference_worker as worker


class FakeClient:
    def __init__(self, endpoint, timeout):
        self.calls = []
        self.mask = np.zeros((4, 4), dtype=bool)
        self.mask[2:, 2:] = True
        self.metadata = dict(state="tracking", generation=1, source="vlm_sam2_cutie", latency_ms=2.0)

    def start(self, rgb, target_text, options):
        self.calls.append(("start", rgb.copy(), target_text, copy.deepcopy(options)))
        return SimpleNamespace(mask=self.mask.copy(), metadata=copy.deepcopy(self.metadata))

    def step(self, rgb, frame_index, force_reacquire=False):
        self.calls.append(("step", rgb.copy(), frame_index, force_reacquire))
        return SimpleNamespace(mask=self.mask.copy(), metadata=copy.deepcopy(self.metadata))

    def close(self):
        self.calls.append(("close",))


class DummyWriter:
    def append_data(self, _image):
        pass

    def close(self):
        pass


class PerceptionBoundaryTests(unittest.TestCase):
    def test_rgb_only_service_boundary_and_policy_uses_perceived_mask(self):
        config = web.normalize_mask_config(dict(mask_source="perception"))
        session = web.PerceptionTrial(config, "011_banana", client_factory=FakeClient)
        lab = web.Lab.__new__(web.Lab)
        lab.mask_config, lab.perception = config, session
        base = SimpleNamespace(cubes=[SimpleNamespace(per_scene_id=np.array(7))], selected_target_idx=0)
        lab.env = SimpleNamespace(unwrapped=base)
        lab.update = lambda **_fields: None
        rgb = np.full((1, 4, 4, 3), 90, dtype=np.uint8)
        segmentation = np.zeros((1, 4, 4, 1), dtype=np.int32)
        segmentation[0, :2, :2, 0] = 7
        raw = dict(sensor_data=dict(base_camera=dict(rgb=rgb, segmentation=segmentation),
                                    hand_camera=dict(rgb=rgb.copy())),
                   agent=dict(qpos=np.zeros((1, 9), dtype=np.float32)))
        observation = lab.observe(raw)
        self.assertEqual(session.client.calls[0][0], "start")
        self.assertEqual(session.client.calls[0][2], "banana")
        self.assertEqual(set(session.client.calls[0][3]), {"loss_mode", "intervention"})
        self.assertEqual(session.client.calls[0][1].shape, (4, 4, 3))
        self.assertEqual(int(observation["gt_mask"].sum()), 4)
        self.assertEqual(observation["perception"]["mask_iou"], 0.0)
        self.assertFalse(observation["target"][:2, :2].any())
        self.assertTrue(observation["target"][2:, 2:].any())
        session.client.mask[:] = False
        session.client.metadata["state"] = "lost"
        lost = lab.observe(raw, frame_index=1)
        self.assertFalse(lost["target"].any())
        self.assertEqual(int(lost["gt_mask"].sum()), 4)
        self.assertEqual(lost["perception"]["state"], "lost")
        self.assertEqual(session.client.calls[-1][0], "step")

    def test_service_error_propagates_instead_of_oracle_fallback(self):
        session = web.PerceptionTrial(dict(mask_source="perception"), "025_mug", client_factory=FakeClient)
        with patch.object(session.client, "start", side_effect=RuntimeError("service unavailable")):
            with self.assertRaisesRegex(RuntimeError, "service unavailable"):
                session.apply(np.zeros((4, 4, 3), dtype=np.uint8), 0)
        self.assertFalse(session.started)

    def test_ambiguous_red_cube_requires_visible_description(self):
        config = web.normalize_mask_config(dict(mask_source="perception"))
        with self.assertRaisesRegex(ValueError, "可见描述"):
            web.target_description("red_cube_1", config)
        config["target_text"] = "leftmost red cube"
        self.assertEqual(web.target_description("red_cube_2", config), "leftmost red cube")

    def test_invalid_oracle_intervention_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "真实感知模式"):
            web.normalize_mask_config(dict(intervention=dict(kind="dropout")))
        with self.assertRaises(ValueError):
            web.normalize_mask_config(dict(perception_url="http://example.com:18830"))

    def test_same_scene_frame_fixes_worker_noise_across_mask_conditions(self):
        lab = web.Lab.__new__(web.Lab)
        lab.scene, lab.policy_port = dict(seed=123), 18833
        payloads = []

        class Response:
            def __init__(self, data):
                self.data = data

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def read(self):
                return json.dumps(self.data).encode()

        def reply(request, **_kwargs):
            data = json.loads(request.data)
            payloads.append(data)
            return Response(dict(actions=np.zeros((2, 8)).tolist(), inference_seconds=0.001,
                                 noise_seed=data["noise_seed"]))

        rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        observation = dict(base=rgb, wrist=rgb.copy(), target=rgb.copy(), state=np.zeros(8),
                           policy_frame_index=10, gt_mask=np.ones((224, 224)), perception={"generation": 1})
        with patch.object(web, "urlopen", side_effect=reply):
            lab.infer(observation)
            observation["target"] = np.full_like(rgb, 100)
            observation["perception"] = {"generation": 3, "state": "lost"}
            lab.infer(observation)
            observation["policy_frame_index"] = 11
            lab.infer(observation)
        seed = payloads[0]["noise_seed"]
        self.assertEqual(seed, payloads[1]["noise_seed"])
        self.assertNotEqual(seed, payloads[2]["noise_seed"])
        self.assertEqual(set(payloads[0]), {"base_rgb", "wrist_rgb", "target_rgb", "state", "noise_seed"})
        left, right = worker.noise_from_seed(seed, (50, 32)), worker.noise_from_seed(seed, (50, 32))
        np.testing.assert_array_equal(left, right)
        self.assertEqual(left.dtype, np.float32)
        self.assertEqual(left.shape, (50, 32))
        for invalid in (True, -1, 2**32, None):
            with self.assertRaises(ValueError):
                worker.noise_from_seed(invalid, (50, 32))


class TrialLoopTests(unittest.TestCase):
    def make_lab(self, folder, config, generation, state=lambda _frame: "tracking"):
        lab = web.Lab.__new__(web.Lab)
        lab.output = Path(folder)
        lab.scene = dict(id="scene", target="011_banana", seed=1, split="train", objects=["011_banana"],
                         heldout_objects=[], allow_heldout=False, used=False)
        lab.mask_config = config
        lab.perception = None
        lab.cancel, lab.reacquire = threading.Event(), threading.Event()
        lab.lock, lab.status = threading.Lock(), dict(results=[])
        lab.update = lambda **fields: lab.status.update(fields)
        lab.checkpoint_path = lambda _name: Path(folder) / "49999"
        lab.ensure_worker = lambda *_args: None
        lab.reset_perception = lambda cfg, _target: setattr(lab, "mask_config", copy.deepcopy(cfg))
        lab.publish = lambda _image: None
        lab.frame = lambda *_args: np.zeros((280, 672, 3), dtype=np.uint8)
        lab.raw_obs = {"frame": 0}
        arm_actions, policy_inputs = [], []

        class Env:
            action_space = SimpleNamespace(low=np.full(8, -1000), high=np.full(8, 1000))
            frame = 0

            def step(self, action):
                arm_actions.append(float(action[0]))
                self.frame += 1
                info = dict(is_target_grasped=False, target_lifted=False,
                            wrong_object_grasped=False, success=False)
                return {"frame": self.frame}, 0, False, False, info

        lab.env = Env()

        def observe(raw, frame_index=0, force_reacquire=False):
            index = raw["frame"]
            current_state = state(index)
            mask = np.ones((4, 4), dtype=bool) if current_state == "tracking" else np.zeros((4, 4), dtype=bool)
            metadata = web.mask_diagnostics(mask, np.ones_like(mask),
                dict(state=current_state, generation=generation(index), source="vlm_sam2_cutie",
                     latency_ms=2.0, events=[]))
            rgb = np.full((4, 4, 3), 90, dtype=np.uint8)
            return dict(base=rgb, wrist=rgb.copy(), target=np.where(mask[..., None], rgb, 0),
                        mask=mask, gt_mask=np.ones_like(mask), state=np.zeros(8, dtype=np.float32),
                        perception=metadata, policy_frame_index=index)

        def infer(observation):
            policy_inputs.append(observation["target"].copy())
            chunk = np.zeros((10, 8), dtype=np.float32)
            chunk[:, 0] = len(policy_inputs) * 100 + np.arange(10)
            return chunk, 0.001

        lab.observe, lab.infer = observe, infer
        return lab, arm_actions, policy_inputs

    def test_chunk_cursor_resets_and_generation_invalidates_old_actions(self):
        config = web.normalize_mask_config(dict(mask_source="perception"))
        with tempfile.TemporaryDirectory() as folder:
            lab, actions, inputs = self.make_lab(folder, config, lambda frame: 1 if frame < 12 else 2)
            with patch.object(web, "MAX_STEPS", 13), patch.object(web.imageio, "get_writer", return_value=DummyWriter()):
                lab.run_trial("49999", 0, config)
            self.assertEqual(actions, list(range(100, 110)) + [200, 201, 300])
            self.assertEqual(len(inputs), 3)
            result = lab.status["results"][-1]
            self.assertEqual(result["mask_source"], "vlm_sam2_cutie")
            self.assertEqual(result["mask_summary"]["generations"], [1, 2])
            self.assertEqual(result["inference_calls"], 3)
            saved = list((Path(folder) / "trials").glob("*/step_metrics.json"))
            traces = json.loads(saved[0].read_text())
            self.assertEqual(len(traces), 13)
            self.assertEqual([row["policy_input_frame"] for row in traces], [0] * 10 + [10, 10, 12])
            self.assertEqual(traces[-1]["policy_input_perception"]["generation"], 2)
            self.assertEqual([row["noise_seed"] for row in traces],
                             [web.sampling_noise_seed(1, row["policy_input_frame"]) for row in traces])
            self.assertEqual(result["sampling_mode"], "scene_frame_seeded_noise")

    def test_pause_on_loss_holds_and_stops_at_bound_without_policy_calls(self):
        config = web.normalize_mask_config(dict(mask_source="perception", loss_mode="pause", max_pause_steps=2))
        with tempfile.TemporaryDirectory() as folder:
            lab, actions, inputs = self.make_lab(folder, config, lambda _frame: 1, state=lambda _frame: "lost")
            with patch.object(web, "MAX_STEPS", 4), patch.object(web.imageio, "get_writer", return_value=DummyWriter()):
                lab.run_trial("49999", 0, config)
            self.assertEqual(actions, [0.0, 0.0])
            self.assertEqual(inputs, [])
            self.assertEqual(lab.status["results"][-1]["stop_reason"], "mask_lost_pause_timeout")
            self.assertEqual(lab.status["results"][-1]["pause_steps"], 2)


if __name__ == "__main__":
    unittest.main()
