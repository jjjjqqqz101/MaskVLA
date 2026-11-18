"""Portable inventory selection/geometry contracts; standard library only."""
import json
import math
from pathlib import Path
import random
import unittest

from task_agent.scene_objects import (CLASSIC_OBJECT_IDS, catalog_public, get_object_spec,
    normalize_object_selection, normalize_scene_spec, resolve_scene_spec)


class FirstSampler:
    def sample(self, population, count): return list(population)[:count]


class ReorderingSampler:
    def sample(self, population, count):
        values = list(population)[:count]
        return values[::-1] if count > 1 else values


class SceneObjectContracts(unittest.TestCase):
    def test_catalog_has40_real_unique_entries_with_separate_training_coverage(self):
        catalog = catalog_public()
        self.assertEqual(len(catalog), 40)
        self.assertEqual(len({row['id'] for row in catalog}), 40)
        self.assertEqual({row['id'] for row in catalog if row['act_trained']}, set(CLASSIC_OBJECT_IDS))
        self.assertEqual(sum(row['pi05_trained'] for row in catalog), 25)
        self.assertEqual(sum(not row['pi05_trained'] for row in catalog), 15)
        for row in catalog:
            self.assertEqual(set(row), {'id', 'name', 'category', 'act_trained', 'pi05_trained', 'available'})
            self.assertTrue(row['available'])
            self.assertIs(type(row['act_trained']), bool)
            self.assertIs(type(row['pi05_trained']), bool)
            self.assertTrue(row['name']); self.assertTrue(row['category'])
        by_id = {row['id']: row for row in catalog}
        self.assertTrue(by_id['054_softball']['act_trained'])
        self.assertTrue(by_id['054_softball']['pi05_trained'])
        self.assertFalse(by_id['056_tennis_ball']['act_trained'])
        self.assertFalse(by_id['056_tennis_ball']['pi05_trained'])
        self.assertEqual(by_id['054_softball']['name'], '垒球')

    def test_catalog_is_portable_and_returned_values_are_independent(self):
        path = Path(__file__).parent/'task_agent/object_catalog.json'
        serialized = path.read_text()
        self.assertNotIn('/home/', serialized)
        self.assertNotIn('/Users/', serialized)
        catalog = catalog_public(); catalog[0]['name'] = 'mutated'
        self.assertNotEqual(catalog_public()[0]['name'], 'mutated')
        cube = get_object_spec('red_cube_1'); cube['color'][0] = .5; cube['bbox_min'][0] = -99
        self.assertEqual(get_object_spec('red_cube_1')['color'], [1, 0, 0, 1])
        self.assertEqual(get_object_spec('red_cube_1')['bbox_min'][0], -.02)
        for value in (None, [], {}, 'missing', '../025_mug'):
            with self.subTest(value=value), self.assertRaises(ValueError): get_object_spec(value)

    def test_scaled_geometry_and_mesh_scale_are_not_applied_twice(self):
        for row in catalog_public():
            obj = get_object_spec(row['id'])
            dimensions = [high-low for low, high in zip(obj['bbox_min'], obj['bbox_max'])]
            for actual, expected in zip(dimensions, obj['dimensions_m']): self.assertAlmostEqual(actual, expected)
            self.assertAlmostEqual(-obj['bbox_min'][2], obj['rest_z_m'])
            self.assertTrue(all(math.isfinite(v) and v > 0 for v in dimensions))
            if obj['kind'] == 'cube':
                self.assertEqual(obj['dimensions_m'], [.04, .04, .04])
                self.assertEqual(obj['default_scale'], 1)
                self.assertEqual(obj['half_size_m'], .02)
                self.assertIsNone(obj['asset_id'])
            else:
                self.assertEqual(obj['kind'], 'ycb')
                self.assertEqual(obj['asset_id'], obj['id'])
        mug = get_object_spec('025_mug')
        self.assertEqual(mug['default_scale'], .7)
        self.assertAlmostEqual(mug['dimensions_m'][0], .10697540850075354*.7)
        self.assertEqual(get_object_spec('054_softball')['default_scale'], .76)
        self.assertEqual(get_object_spec('055_baseball')['default_scale'], .58)

    def test_default_spec_preserves_original_eight_order_and_independence(self):
        result = normalize_scene_spec()
        self.assertEqual(result['version'], 1)
        self.assertEqual(result['preset'], 'classic8')
        self.assertEqual([obj['id'] for obj in result['objects']], list(CLASSIC_OBJECT_IDS))
        self.assertEqual(normalize_scene_spec(result), result)
        result['objects'][0]['scale'] = 9
        self.assertEqual(normalize_scene_spec()['objects'][0]['scale'], 1)
        self.assertEqual(normalize_object_selection(), {'mode': 'classic'})
        self.assertEqual(resolve_scene_spec(None), normalize_scene_spec())

    def test_manual_choice_keeps_order_and_catalog_scales(self):
        ids = ['056_tennis_ball', '025_mug', 'red_cube_2']
        result = resolve_scene_spec(dict(mode='manual', object_ids=ids))
        self.assertEqual(result['preset'], 'custom')
        self.assertEqual(result['objects'], [dict(id=i, scale=get_object_spec(i)['default_scale']) for i in ids])
        self.assertEqual(normalize_scene_spec(result), result)
        result['objects'].reverse()
        self.assertEqual([r['id'] for r in resolve_scene_spec(dict(mode='manual', object_ids=ids))['objects']], ids)

    def test_invalid_selection_modes_ids_counts_and_extra_fields_are_rejected(self):
        invalid = [[], 'classic', {}, {'mode': 'unknown'}, {'mode': 'classic', 'count': 8},
            {'mode': 'manual', 'object_ids': []}, {'mode': 'manual', 'object_ids': ['red_cube_1']*2},
            {'mode': 'manual', 'object_ids': ['missing']}, {'mode': 'manual', 'object_ids': 'red_cube_1'},
            {'mode': 'manual', 'object_ids': list(CLASSIC_OBJECT_IDS)+['013_apple']},
            {'mode': 'random', 'count': 2, 'pool': 'heldout'},
            {'mode': 'random', 'count': 2, 'pool': 'train', 'target_idx': 1}]
        invalid += [dict(mode='random', count=x, pool='all') for x in (True, False, 0, 9, 1.5, '2', None)]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError): normalize_object_selection(value)

    def test_invalid_resolved_specs_cannot_change_scale_or_rename_classic_layout(self):
        custom = resolve_scene_spec(dict(mode='manual', object_ids=['025_mug']))
        invalid = [[], {}, dict(custom, version=True), dict(custom, version=1.0), dict(custom, version=2),
            dict(custom, preset='unknown'), dict(custom, object_id='025_mug'),
            dict(custom, objects=[]), dict(custom, preset='classic8'),
            dict(custom, objects=[{'id': '025_mug'}]),
            dict(custom, objects=[{'id': '025_mug', 'scale': .7, 'world_pose': [0, 0, 0]}]),
            dict(custom, objects=custom['objects']*2)]
        invalid += [dict(custom, objects=[dict(id='025_mug', scale=x)]) for x in
                    (True, False, 1, 0, .5, float('nan'), float('inf'), '0.7', None)]
        swapped = normalize_scene_spec(); swapped['objects'][0], swapped['objects'][1] = swapped['objects'][1], swapped['objects'][0]
        invalid.append(swapped)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError): normalize_scene_spec(value)

    def test_random_pool_selection_is_bounded_unique_and_reproducible_with_supplied_rng(self):
        train_ids = {row['id'] for row in catalog_public() if row['pi05_trained']}
        all_ids = {row['id'] for row in catalog_public()}
        for pool in ('train', 'all'):
            for count in (1, 8):
                choice = dict(mode='random', pool=pool, count=count)
                first = resolve_scene_spec(choice, rng=random.Random(17))
                second = resolve_scene_spec(choice, rng=random.Random(17))
                self.assertEqual(first, second)
                ids = [row['id'] for row in first['objects']]
                self.assertEqual(len(ids), count)
                self.assertEqual(len(set(ids)), count)
                self.assertTrue(set(ids) <= (train_ids if pool == 'train' else all_ids))

    def test_random_refresh_changes_inventory_even_if_rng_only_reorders_previous_set(self):
        old = normalize_scene_spec()
        fresh = resolve_scene_spec(dict(mode='random', count=8, pool='train'),
                                   previous_spec=old, rng=ReorderingSampler())
        self.assertNotEqual({r['id'] for r in fresh['objects']}, {r['id'] for r in old['objects']})
        self.assertEqual(len({r['id'] for r in fresh['objects']}), 8)
        self.assertEqual(old, normalize_scene_spec())

    def test_random_single_red_cube_cannot_refresh_to_identical_appearance_other_red_id(self):
        old = resolve_scene_spec(dict(mode='manual', object_ids=['red_cube_2']))
        fresh = resolve_scene_spec(dict(mode='random', count=1, pool='train'),
                                   previous_spec=old, rng=FirstSampler())
        self.assertNotIn(fresh['objects'][0]['id'], ('red_cube_1', 'red_cube_2'))

    def test_frozen_spec_json_roundtrip_does_not_reroll_or_reorder_inventory(self):
        resolved = resolve_scene_spec(dict(mode='random', count=6, pool='all'), rng=random.Random(91))
        restored = normalize_scene_spec(json.loads(json.dumps(resolved)))
        self.assertEqual(restored, resolved)
        self.assertEqual(normalize_scene_spec(restored), resolved)


if __name__ == '__main__': unittest.main()
