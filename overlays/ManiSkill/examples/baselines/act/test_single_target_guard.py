"""RGB-mask geometry regressions; no model, simulator, GT or GPU dependency."""
import json
import unittest

import numpy as np

from perception_mask.client import decode_png, encode_rgb
from perception_mask.core import TrackingSession, options_from_dict
from perception_mask.server import Application
from perception_mask.single_target_guard import SingleTargetGuard


def rectangle(box, shape=(64, 64)):
    mask = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return mask


def occlusion_sequence():
    seed = rectangle([20, 20, 32, 33])
    seed[20, 20:27] = False
    fragment = rectangle([24, 28, 28, 32])
    fragment[28, 24:27] = False
    restored = rectangle([23, 16, 35, 30])
    restored[16, 28:35] = False
    masks = [seed, rectangle([20, 21, 32, 32]), rectangle([21, 22, 31, 32]),
             rectangle([22, 24, 30, 32]), rectangle([23, 26, 29, 32]),
             rectangle([24, 27, 28, 32]), fragment, restored]
    return masks


class SingleTargetGeometry(unittest.TestCase):
    def test_seed_chooses_unique_tight_box_supported_component_not_largest(self):
        target = rectangle([10, 10, 18, 18])
        mask = target | rectangle([40, 40, 58, 58])
        result = SingleTargetGuard().seed(mask, [10, 10, 18, 18])
        self.assertTrue(result.accepted)
        np.testing.assert_array_equal(result.mask, target)
        self.assertEqual(result.metadata['discarded_pixels'], 324)

    def test_two_significant_components_inside_cue_are_rejected_not_ranked(self):
        mask = rectangle([10, 10, 18, 18]) | rectangle([23, 10, 31, 18])
        result = SingleTargetGuard().seed(mask, [8, 8, 35, 20])
        self.assertFalse(result.accepted)
        self.assertFalse(result.mask.any())
        self.assertEqual(result.metadata['reason'], 'ambiguous_seed_components')

    def test_seed_cannot_fall_back_to_background_or_largest_unrelated_object(self):
        for mask in (rectangle([0, 0, 35, 35]), rectangle([40, 40, 58, 58])):
            with self.subTest(pixels=int(mask.sum())):
                result = SingleTargetGuard().seed(mask, [10, 10, 18, 18])
                self.assertFalse(result.accepted)
                self.assertFalse(result.mask.any())

    def test_continuous_movement_is_accepted_and_rejection_keeps_trusted_history(self):
        old = rectangle([10, 10, 20, 20])
        guard = SingleTargetGuard()
        guard.trust(old)
        shifted = rectangle([12, 10, 22, 20])
        self.assertTrue(guard.prediction(shifted).accepted)
        distant = rectangle([42, 42, 52, 52])
        rejected = guard.prediction(distant)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.metadata['reason'], 'spatial_identity_jump')
        np.testing.assert_array_equal(guard.last_trusted, old)
        self.assertTrue(guard.prediction(old).accepted)

    def test_real_two_cube_component_distribution_is_rejected(self):
        # Reconstruct component sizes/bounds from a prior Cutie output, not GT:
        #58 pixels [90,100,98,108] plus51 pixels [108,118,115,126].
        old = rectangle([90, 100, 98, 108], (224, 224))
        old[100, 90:96] = False
        other = rectangle([108, 118, 115, 126], (224, 224))
        other[118, 108:113] = False
        self.assertEqual((int(old.sum()), int(other.sum())), (58, 51))
        guard = SingleTargetGuard()
        guard.trust(old)
        result = guard.prediction(old | other)
        self.assertFalse(result.accepted)
        self.assertEqual(result.metadata['reason'], 'multiple_objects_or_discontinuous_fragments')
        self.assertFalse(result.mask.any())
        np.testing.assert_array_equal(guard.last_trusted, old)

    def test_occlusion_fragments_inside_trusted_silhouette_are_compatible(self):
        old = rectangle([20, 20, 30, 30])
        fragments = rectangle([20, 20, 23, 30]) | rectangle([27, 20, 30, 30])
        guard = SingleTargetGuard()
        guard.trust(old)
        result = guard.prediction(fragments)
        self.assertTrue(result.accepted)
        self.assertEqual(result.metadata['reason'], 'trusted_silhouette_fragments')
        np.testing.assert_array_equal(result.mask, fragments)

    def test_connected_area_merge_and_tiny_remote_noise_are_distinguished(self):
        old = rectangle([20, 20, 30, 30])
        guard = SingleTargetGuard()
        guard.trust(old)
        merged = guard.prediction(rectangle([20, 20, 40, 30]))
        self.assertFalse(merged.accepted)
        self.assertEqual(merged.metadata['reason'], 'area_changed_abruptly')
        noisy = guard.prediction(old | rectangle([50, 50, 52, 52]))
        self.assertTrue(noisy.accepted)
        np.testing.assert_array_equal(noisy.mask, old)

    def test_gradual_occlusion_keeps_fuller_anchor_and_recovers149_to13_to161(self):
        masks = occlusion_sequence()
        self.assertEqual([int(m.sum()) for m in masks], [149, 132, 100, 64, 36, 20, 13, 161])
        guard = SingleTargetGuard()
        guard.trust(masks[0])
        for mask in masks[1:-1]:
            checked = guard.prediction(mask)
            self.assertTrue(checked.accepted, checked.metadata)
            guard.trust(checked.mask)
        self.assertEqual(int(guard.last_trusted.sum()), 13)
        self.assertEqual(int(guard.fuller_anchor.sum()), 132)
        self.assertEqual(guard.peak_trusted_pixels, 149)
        restored = guard.prediction(masks[-1])
        self.assertTrue(restored.accepted, restored.metadata)
        self.assertEqual(restored.metadata['reference_used'], 'fuller_anchor')
        self.assertAlmostEqual(restored.metadata['area_ratio'], 161/13)
        self.assertLess(restored.metadata['centroid_shift_pixels'], 8)
        guard.trust(restored.mask)
        self.assertEqual(guard.peak_trusted_pixels, 161)
        np.testing.assert_array_equal(guard.fuller_anchor, masks[-1])

    def test_occluded_reference_cannot_rescue_far_same_size_or_multiple_objects(self):
        masks = occlusion_sequence()
        far = rectangle([45, 45, 57, 59])
        far[45, 50:57] = False
        self.assertEqual(int(far.sum()), 161)
        for candidate in (far, masks[-1] | rectangle([45, 45, 51, 51])):
            with self.subTest(components='far' if candidate.sum() == 161 else 'multiple'):
                guard = SingleTargetGuard()
                guard.trust(masks[0])
                for mask in masks[1:-1]:
                    checked = guard.prediction(mask)
                    self.assertTrue(checked.accepted)
                    guard.trust(checked.mask)
                previous = guard.last_trusted.copy()
                anchor = guard.fuller_anchor.copy()
                rejected = guard.prediction(candidate)
                self.assertFalse(rejected.accepted)
                self.assertFalse(rejected.mask.any())
                np.testing.assert_array_equal(guard.last_trusted, previous)
                np.testing.assert_array_equal(guard.fuller_anchor, anchor)
                self.assertEqual(guard.peak_trusted_pixels, 149)
                self.assertTrue(guard.prediction(masks[-1]).accepted)


class FakeLocator:
    def __init__(self, box=None):
        self.box = box or [10, 10, 20, 20]
        self.calls = 0

    def locate(self, rgb, text):
        self.calls += 1
        return dict(found=True, bbox_xyxy=self.box, coordinate_space='pixels')


class FakeSegmenter:
    def __init__(self, masks):
        self.masks = list(masks)
        self.boxes = []

    def segment(self, rgb, box):
        self.boxes.append(list(box))
        return self.masks[min(len(self.boxes)-1, len(self.masks)-1)].copy(), .95


class FakeTracker:
    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.accepted = self.rejected = 0
        self.closed = False
        self.initial = None

    def initialize(self, image, mask): self.initial = mask.copy()
    def predict(self, image): return self.predictions.pop(0)
    def accept(self): self.accepted += 1
    def reject(self): self.rejected += 1
    def close(self): self.closed = True


class TrackingGuardContracts(unittest.TestCase):
    def session(self, seed, predictions=(), options=None, fresh_seed=None, locator_box=None):
        self.rgb = np.zeros((*seed.shape, 3), dtype=np.uint8)
        self.trackers = []
        def factory():
            tracker = FakeTracker(predictions)
            self.trackers.append(tracker)
            return tracker
        locator = FakeLocator(locator_box)
        segmenter = FakeSegmenter([seed, fresh_seed] if fresh_seed is not None else [seed])
        cfg = dict(single_target=True, single_target_bbox=[10, 10, 20, 20], loss_mode='observe')
        if options is not None:
            cfg = options
        session = TrackingSession(locator, segmenter, factory, 'one visible cube', cfg,
            initial_localization=dict(found=True, bbox_xyxy=[6, 6, 24, 24], coordinate_space='pixels'))
        self.addCleanup(session.close)
        return session, locator, segmenter

    def test_rejected_tracking_output_is_lost_rejects_memory_and_can_recover_trusted_target(self):
        old = rectangle([10, 10, 20, 20])
        multi = old | rectangle([40, 40, 50, 50])
        session, _, _ = self.session(old, [(multi, .98), (old, .98)])
        self.assertEqual(session.process(self.rgb, 0).metadata['state'], 'tracking')
        lost = session.process(self.rgb, 1)
        self.assertEqual(lost.metadata['state'], 'lost')
        self.assertTrue(lost.metadata['natural_lost'])
        self.assertFalse(lost.mask.any())
        self.assertEqual(self.trackers[0].rejected, 1)
        self.assertEqual(self.trackers[0].accepted, 0)
        np.testing.assert_array_equal(session.last_good_mask, old)
        np.testing.assert_array_equal(session.single_target_guard.last_trusted, old)
        recovered = session.process(self.rgb, 2)
        self.assertEqual(recovered.metadata['state'], 'tracking')
        self.assertEqual(self.trackers[0].accepted, 1)
        json.dumps(recovered.metadata, allow_nan=False)

    def test_ambiguous_seed_never_initializes_tracker_or_generation(self):
        seed = rectangle([10, 10, 14, 20]) | rectangle([16, 10, 20, 20])
        session, _, _ = self.session(seed)
        result = session.process(self.rgb, 0)
        self.assertFalse(result.mask.any())
        self.assertEqual(result.metadata['state'], 'lost')
        self.assertEqual(result.metadata['single_target_guard']['reason'], 'ambiguous_seed_components')
        self.assertEqual(result.metadata['generation'], 0)
        self.assertEqual(self.trackers, [])

    def test_low_quality_candidate_cannot_replace_trusted_identity(self):
        old = rectangle([10, 10, 20, 20])
        wrong = rectangle([40, 40, 50, 50])
        session, _, _ = self.session(old, [(wrong, .1), (old, .95)])
        session.process(self.rgb, 0)
        lost = session.process(self.rgb, 1)
        self.assertEqual(lost.metadata['state'], 'lost')
        self.assertEqual(lost.metadata['single_target_guard']['reason'], 'tracker_mask_quality_rejected')
        self.assertEqual(self.trackers[0].rejected, 1)
        np.testing.assert_array_equal(session.single_target_guard.last_trusted, old)
        self.assertEqual(session.process(self.rgb, 2).metadata['state'], 'tracking')

    def test_padded_sam_prompt_and_tight_guard_cue_remain_distinct(self):
        target = rectangle([10, 10, 20, 20])
        neighbor = rectangle([21, 10, 24, 20])
        session, _, segmenter = self.session(target | neighbor)
        result = session.process(self.rgb, 0)
        self.assertEqual(segmenter.boxes, [[6., 6., 24., 24.]])
        np.testing.assert_array_equal(result.mask, target)
        np.testing.assert_array_equal(self.trackers[0].initial, target)
        self.assertEqual(result.metadata['single_target_guard']['tight_bbox_xyxy'], [10., 10., 20., 20.])

    def test_reacquisition_uses_new_grounder_box_and_closes_old_tracker(self):
        old = rectangle([10, 10, 20, 20])
        fresh = rectangle([40, 40, 48, 48])
        session, locator, _ = self.session(old, fresh_seed=fresh, locator_box=[40, 40, 48, 48])
        session.process(self.rgb, 0)
        result = session.process(self.rgb, 1, force_reacquire=True)
        self.assertEqual(locator.calls, 1)
        self.assertTrue(self.trackers[0].closed)
        self.assertEqual(result.metadata['generation'], 2)
        self.assertEqual(result.metadata['single_target_guard']['tight_bbox_xyxy'], [40., 40., 48., 48.])
        np.testing.assert_array_equal(result.mask, fresh)
        self.assertEqual(session.single_target_guard.peak_trusted_pixels, 64)
        np.testing.assert_array_equal(session.single_target_guard.fuller_anchor, fresh)

    def test_tracker_accepts_nearby_reappearance_after_progressive_occlusion(self):
        masks = occlusion_sequence()
        session, _, _ = self.session(masks[0], [(mask, .978) for mask in masks[1:]],
            options=dict(single_target=True, single_target_bbox=[20, 20, 32, 33]))
        self.assertEqual(session.process(self.rgb, 0).metadata['state'], 'tracking')
        for frame in range(1, len(masks)):
            result = session.process(self.rgb, frame)
            self.assertEqual(result.metadata['state'], 'tracking', result.metadata)
        self.assertEqual(result.metadata['single_target_guard']['reason'], 'recovered_single_target_after_occlusion')
        self.assertEqual(self.trackers[0].accepted, len(masks)-1)
        self.assertEqual(self.trackers[0].rejected, 0)
        np.testing.assert_array_equal(session.last_good_mask, masks[-1])

    def test_far_reappearance_after_occlusion_rejects_memory_without_poisoning_anchor(self):
        masks = occlusion_sequence()
        far = rectangle([45, 45, 57, 59])
        far[45, 50:57] = False
        predictions = [(mask, .978) for mask in masks[1:-1]]+[(far, .978), (masks[-1], .978)]
        session, _, _ = self.session(masks[0], predictions,
            options=dict(single_target=True, single_target_bbox=[20, 20, 32, 33]))
        for frame in range(len(masks)-1):
            self.assertEqual(session.process(self.rgb, frame).metadata['state'], 'tracking')
        rejected = session.process(self.rgb, len(masks)-1)
        self.assertEqual(rejected.metadata['state'], 'lost')
        self.assertFalse(rejected.mask.any())
        self.assertEqual(self.trackers[0].rejected, 1)
        self.assertEqual(int(session.single_target_guard.fuller_anchor.sum()), 132)
        recovered = session.process(self.rgb, len(masks))
        self.assertEqual(recovered.metadata['state'], 'tracking')
        np.testing.assert_array_equal(recovered.mask, masks[-1])

    def test_old_api_default_preserves_masks_without_enabling_guard(self):
        target = rectangle([10, 10, 20, 20])
        multi = target | rectangle([40, 40, 50, 50])
        segmenter = FakeSegmenter([multi])
        trackers = []
        def factory():
            tracker = FakeTracker([])
            trackers.append(tracker)
            return tracker
        app = Application(FakeLocator(), segmenter, factory)
        self.addCleanup(lambda: [session.close() for session in app.sessions.values()])
        response = app.request('/v1/start', dict(image_png=encode_rgb(np.zeros((64, 64, 3), np.uint8)),
                                                 target_text='old experiment target'))
        np.testing.assert_array_equal(decode_png(response['mask_png'], mask=True), multi)
        np.testing.assert_array_equal(trackers[0].initial, multi)
        self.assertNotIn('single_target_guard', response['metadata'])

    def test_guard_options_are_explicit_and_boxes_strictly_validated(self):
        self.assertFalse(options_from_dict({})['single_target'])
        invalid = [dict(single_target=1), dict(single_target='true'),
            dict(single_target=False, single_target_bbox=[1, 1, 10, 10])]
        invalid += [dict(single_target=True, single_target_bbox=box) for box in (
            [True, 1, 10, 10], [1, 1, float('nan'), 10], [1, 1, float('inf'), 10],
            [-1, 1, 10, 10], [10, 1, 1, 10], [1, 1, 2, 10], [1, 2, 3])]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                options_from_dict(options)


if __name__ == '__main__':
    unittest.main()
