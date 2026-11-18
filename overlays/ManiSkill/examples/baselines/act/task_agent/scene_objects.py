"""Portable scene inventory contracts, without robot/model imports.

Training flags describe this project's fine-tuning targets, not guaranteed grasp
success or what the pretrained model has seen. Catalog bounds/dimensions are
already scaled by default_scale; only mesh loading should apply that scale.
"""
import copy
import json
import math
from pathlib import Path
import random


CLASSIC_OBJECT_IDS = (
    'red_cube_1', 'red_cube_2', 'blue_cube', '003_cracker_box',
    '025_mug', '011_banana', '054_softball', '055_baseball',
)
_DATA = json.loads(Path(__file__).with_name('object_catalog.json').read_text())
_OBJECTS = {row['id']: row for row in _DATA['objects']}
_PUBLIC_FIELDS = ('id', 'name', 'category', 'act_trained', 'pi05_trained', 'available')


def get_object_spec(object_id):
    """Return independent asset metadata; AABB and dimensions are scaled meters."""
    if not isinstance(object_id, str) or object_id not in _OBJECTS:
        raise ValueError('Unknown scene object ID')
    if not _OBJECTS[object_id]['available']:
        raise ValueError('Scene object asset is unavailable')
    return copy.deepcopy(_OBJECTS[object_id])


def catalog_public():
    return [{key: copy.deepcopy(row[key]) for key in _PUBLIC_FIELDS} for row in _OBJECTS.values()]


def _ids(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError('Choose between one and eight scene objects')
    result = []
    for object_id in value:
        get_object_spec(object_id)
        if object_id in result:
            raise ValueError('Scene object IDs must be unique')
        result.append(object_id)
    return result


def _resolved(ids, preset):
    return dict(version=1, preset=preset,
                objects=[dict(id=object_id, scale=get_object_spec(object_id)['default_scale']) for object_id in ids])


def _appearance_signature(ids):
    return sorted(_OBJECTS[object_id]['appearance_group'] for object_id in ids)


def normalize_scene_spec(value=None):
    """Validate a frozen, ordered inventory; None means the original eight."""
    if value is None:
        return _resolved(CLASSIC_OBJECT_IDS, 'classic8')
    if not isinstance(value, dict) or set(value) != {'version', 'preset', 'objects'}:
        raise ValueError('Scene spec must contain version, preset and objects only')
    if type(value['version']) is not int or value['version'] != 1:
        raise ValueError('Unsupported scene spec version')
    if value['preset'] not in ('classic8', 'custom'):
        raise ValueError('Scene preset must be classic8 or custom')
    rows = value['objects']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        raise ValueError('Scene spec requires one to eight objects')
    ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'id', 'scale'}:
            raise ValueError('Each resolved object must contain id and scale only')
        spec = get_object_spec(row['id'])
        scale = row['scale']
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or float(scale) != spec['default_scale']):
            raise ValueError('Scene objects must use their catalog default scale')
        ids.append(row['id'])
    _ids(ids)
    if value['preset'] == 'classic8' and tuple(ids) != CLASSIC_OBJECT_IDS:
        raise ValueError('Classic inventory must preserve the original eight-object order')
    return _resolved(ids, value['preset'])


def normalize_object_selection(value=None):
    """Validate the user choice before resolving random contents exactly once."""
    if value is None:
        return dict(mode='classic')
    if not isinstance(value, dict):
        raise ValueError('Object selection must be an object')
    mode = value.get('mode')
    if mode == 'classic' and set(value) == {'mode'}:
        return dict(mode='classic')
    if mode == 'manual' and set(value) == {'mode', 'object_ids'}:
        return dict(mode='manual', object_ids=_ids(value['object_ids']))
    if mode == 'random' and set(value) == {'mode', 'count', 'pool'}:
        count = value['count']
        if type(count) is not int or not 1 <= count <= 8:
            raise ValueError('Random object count must be an integer from one to eight')
        if value['pool'] not in ('train', 'all'):
            raise ValueError('Random object pool must be train or all')
        return dict(mode='random', count=count, pool=value['pool'])
    raise ValueError('Invalid object selection mode or fields')


def resolve_scene_spec(selection, previous_spec=None, rng=None):
    """Freeze selected IDs/order/scales; a random refresh changes composition."""
    choice = normalize_object_selection(selection)
    if choice['mode'] == 'classic':
        return normalize_scene_spec()
    if choice['mode'] == 'manual':
        return _resolved(choice['object_ids'], 'custom')
    generator = rng if rng is not None else random.SystemRandom()
    if not callable(getattr(generator, 'sample', None)):
        raise ValueError('Random resolver needs a random generator with sample()')
    pool = [row['id'] for row in _OBJECTS.values()
            if row['available'] and (choice['pool'] == 'all' or row['pi05_trained'])]
    selected = list(generator.sample(pool, choice['count']))
    if previous_spec is not None:
        previous = normalize_scene_spec(previous_spec)
        old_ids = {row['id'] for row in previous['objects']}
        if _appearance_signature(selected) == _appearance_signature(old_ids):
            index = generator.sample(range(len(selected)), 1)[0]
            group = _OBJECTS[selected[index]]['appearance_group']
            alternatives = [object_id for object_id in pool if object_id not in old_ids
                            and object_id not in selected and _OBJECTS[object_id]['appearance_group'] != group]
            if not alternatives:
                raise ValueError('Object pool has no different composition to refresh')
            selected[index] = generator.sample(alternatives, 1)[0]
    return normalize_scene_spec(_resolved(selected, 'custom'))
