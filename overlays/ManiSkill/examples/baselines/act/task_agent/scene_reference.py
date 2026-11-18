"""Strict, dependency-free references to an earlier scene's initial RGB."""
import re


RGB_HASH_KEYS=('base_rgb','wrist_rgb','overview_rgb')


def validate_scene_reference(value,expected_seed=None):
    if not isinstance(value,dict) or set(value)!={'source_scene_id','seed','rgb_sha256'}:
        raise ValueError('Scene reference must contain source_scene_id, seed and rgb_sha256 only')
    source=value['source_scene_id']
    if not isinstance(source,str) or not re.fullmatch(r'task_agent_[A-Za-z0-9_-]{1,100}',source):
        raise ValueError('Invalid source_scene_id')
    seed=value['seed']
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Invalid reference seed')
    if expected_seed is not None:
        if type(expected_seed) is not int or not 0<=expected_seed<2**31:raise ValueError('Invalid expected seed')
        if seed!=expected_seed:raise ValueError('Reference seed does not match the requested scene seed')
    hashes=value['rgb_sha256']
    if not isinstance(hashes,dict) or set(hashes)!=set(RGB_HASH_KEYS):raise ValueError('Reference requires exactly three RGB hashes')
    if any(not isinstance(hashes[name],str) or not re.fullmatch(r'[0-9a-fA-F]{64}',hashes[name]) for name in RGB_HASH_KEYS):
        raise ValueError('Reference RGB hashes must be 64 hexadecimal characters')
    return dict(source_scene_id=source,seed=seed,rgb_sha256={name:hashes[name].lower() for name in RGB_HASH_KEYS})
