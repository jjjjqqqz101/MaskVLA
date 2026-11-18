"""Mock visual grounding contracts; no network, credentials or simulator."""
import unittest

import numpy as np

from task_agent.vlm import TaskVLM


class GroundingContracts(unittest.TestCase):
    def snapshot(self, width=160, height=80):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
        rgb[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
        return dict(base_rgb=rgb, frame_id=17, captured_at=5.)

    def vlm(self, responses):
        vlm = TaskVLM.__new__(TaskVLM)
        vlm.instruction = '按顺序收拾球类'
        calls = []
        def request(purpose, system, prompt, snapshots, images):
            index = len(calls)
            calls.append(dict(purpose=purpose, system=system, prompt=prompt,
                              snapshots=snapshots, images=images))
            self.assertLess(index, len(responses), 'Unexpected extra model call')
            return responses[index], 'mock_request_%d' % (index + 1)
        vlm._request = request
        return vlm, calls

    def coarse(self, box, description='red ball', coordinate_space='pixels'):
        return dict(decision='pick', description=description, bbox_xyxy=box,
                    coordinate_space=coordinate_space, reason='Visible selected ball')

    def refined(self, box):
        return dict(found=True, bbox_xyxy=box, coordinate_space='normalized_1000',
                    reason='Tight visible ball silhouette')

    def test_non_square_edge_crop_maps_local_box_back_to_original_frame(self):
        snapshot = self.snapshot()
        original = snapshot['base_rgb'].copy()
        vlm, calls = self.vlm([self.coarse([151, 60, 159, 68]),
                               self.refined([700, 550, 960, 760])])
        result = vlm.target(snapshot, {}, [])
        self.assertEqual([c['purpose'] for c in calls], ['subtask_bbox', 'subtask_bbox_refine'])
        self.assertEqual(len(calls[1]['images']), 1)
        self.assertEqual(calls[1]['images'][0].size, (504, 672))
        self.assertEqual(result['crop_xyxy'], [127, 36, 160, 80])
        self.assertEqual(result['coarse_bbox_xyxy'], [151., 60., 159., 68.])
        np.testing.assert_allclose(result['bbox_xyxy'], [150.1, 60.2, 158.68, 69.44])
        self.assertEqual(result['request_id'], 'mock_request_2')
        self.assertEqual(result['refined_request_id'], 'mock_request_2')
        self.assertEqual(result['coarse_request_id'], 'mock_request_1')
        self.assertEqual(result['source_frame_id'], 17)
        self.assertEqual(result['description'], 'red ball')
        np.testing.assert_array_equal(snapshot['base_rgb'], original)

    def test_context_margin_scales_with_coarse_object_and_keeps_crop_aspect_ratio(self):
        vlm, calls = self.vlm([self.coarse([100, 110, 130, 120]),
                               self.refined([400, 400, 600, 600])])
        result = vlm.target(self.snapshot(224, 224), {}, [])
        self.assertEqual(result['crop_xyxy'], [40, 50, 190, 180])
        self.assertEqual(calls[1]['images'][0].size, (672, 582))
        np.testing.assert_allclose(result['bbox_xyxy'], [100, 102, 130, 128])

    def test_coarse_normalized_grid_and_refinement_grid_have_separate_origins(self):
        vlm, _ = self.vlm([self.coarse([943.75, 750, 993.75, 850], coordinate_space='normalized_1000'),
                            self.refined([700, 550, 960, 760])])
        result = vlm.target(self.snapshot(), {}, [])
        self.assertEqual(result['coarse_bbox_xyxy'], [151., 60., 159., 68.])
        np.testing.assert_allclose(result['bbox_xyxy'], [150.1, 60.2, 158.68, 69.44])

    def test_complete_or_uncertain_selection_does_not_make_refinement_call(self):
        for decision in ('complete', 'uncertain'):
            with self.subTest(decision=decision):
                vlm, calls = self.vlm([dict(decision=decision, reason='No visible remaining target')])
                result = vlm.target(self.snapshot(), {}, [])
                self.assertEqual(result['decision'], decision)
                self.assertEqual(len(calls), 1)
                self.assertNotIn('bbox_xyxy', result)

    def test_target_absent_in_crop_is_uncertain_without_coarse_box_fallback(self):
        vlm, calls = self.vlm([self.coarse([151, 60, 159, 68]),
                               dict(found=False, bbox_xyxy=None, reason='Only table visible in this crop')])
        result = vlm.target(self.snapshot(), {}, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['decision'], 'uncertain')
        self.assertNotIn('bbox_xyxy', result)
        self.assertEqual(result['reason'], 'Only table visible in this crop')
        self.assertEqual(result['description'], 'red ball')
        self.assertEqual(result['coarse_bbox_xyxy'], [151., 60., 159., 68.])

    def test_invalid_fine_response_cannot_supply_sam_with_the_coarse_box(self):
        invalid = [self.refined([0, 0, 1001, 800]), self.refined([900, 500, 100, 700]),
                   self.refined([10, 10, float('nan'), 500]),
                   dict(self.refined([100, 100, 900, 900]), found='true'),
                   dict(self.refined([100, 100, 900, 900]), coordinate_space='pixels')]
        for response in invalid:
            with self.subTest(response=response):
                vlm, _ = self.vlm([self.coarse([151, 60, 159, 68]), response])
                result = vlm.target(self.snapshot(), {}, [])
                self.assertEqual(result['decision'], 'uncertain')
                self.assertNotIn('bbox_xyxy', result)

    def test_refinement_must_match_plan_and_retry_identity_not_only_coarse_choice(self):
        plan = {'steps': [{'description': 'large red ball'}, {'description': 'small white ball'}]}
        hint = {'description': 'large red ball', 'failure_reason': 'mask_target_mismatch'}
        vlm, calls = self.vlm([self.coarse([151, 60, 159, 68], description='small white ball'),
                               dict(found=False, reason='Coarse selected a different object')])
        result = vlm.target(self.snapshot(), plan, [], hint)
        refinement = calls[1]
        self.assertEqual(refinement['prompt']['planned_or_retry_target_description'], 'large red ball')
        self.assertEqual(refinement['prompt']['coarse_target_description'], 'small white ball')
        self.assertEqual(refinement['prompt']['plan'], plan)
        self.assertEqual(refinement['prompt']['pending_hint'], hint)
        self.assertIn('return found=false even if that other object is visible', refinement['system'])
        self.assertEqual(result['decision'], 'uncertain')
        self.assertEqual(result['description'], 'large red ball')
        self.assertNotIn('bbox_xyxy', result)

    def test_success_preserves_intended_description_and_ignores_new_model_identity(self):
        response = self.refined([700, 550, 960, 760])
        response['description'] = 'another object invented by refinement'
        vlm, _ = self.vlm([self.coarse([151, 60, 159, 68], description='ball at moved location'), response])
        result = vlm.target(self.snapshot(), {}, [], {'description': 'original selected red ball'})
        self.assertEqual(result['decision'], 'pick')
        self.assertEqual(result['description'], 'original selected red ball')


if __name__ == '__main__':
    unittest.main()
