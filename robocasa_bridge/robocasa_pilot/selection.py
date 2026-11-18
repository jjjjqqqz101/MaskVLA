"""Five frozen official object identities for pilot_v1; no implicit writes."""
from copy import deepcopy
from pathlib import Path
import hashlib

from robocasa_bridge.paths import ASSETS as ASSET_ROOT
FIXED_REQUESTS = [
    {"catalog_id": "objaverse:apple", "instance_id": "a6eae51a3a84ead3fc16"},
    {"catalog_id": "objaverse:banana", "instance_id": "3bcc2bbb7561ade82efa"},
    {"catalog_id": "objaverse:can", "instance_id": "f325cc9bd28f748fc109"},
    {"catalog_id": "objaverse:bar_soap", "instance_id": "d805794bdce6e417fd24"},
    {"catalog_id": "aigen:mug", "instance_id": "07995b7b05ee078c6632"},
]
FROZEN_MODELS = {
    "apple": ("objects/objaverse/apple/apple_20/model.xml", "f30c4d89fb4a54cfa7b3e387185e40b58b43aa61b79fe1db49dc2a07cb1c2b28"),
    "banana": ("objects/objaverse/banana/banana_9/model.xml", "d3554a874c2dbd9e532f123bb8d9fd66d532964ea31ee4c20ee180a52186c853"),
    "can": ("objects/objaverse/can/can_11/model.xml", "9c4277255f88ddaa44d2835c0338513db97b50423c31e1cf7d9a719f0563f1dd"),
    "bar_soap": ("objects/objaverse/bar_soap/bar_soap_3/model.xml", "9ee8fe5573c3ad86f51f5d8612bb083c6a43d00b086c62d5ce301b1e6973ba95"),
    "mug": ("objects/aigen_objs/mug/mug_2/model.xml", "42f217330b14677af3ea92574daebb54f8fe7a765ed6c7514d5b3c0a30ef3a37"),
}
OBJECT_NAMES = tuple(FROZEN_MODELS)


def selection_requests(names=None):
    """Return exact catalog requests, suitable for bridge resolve_selection."""
    wanted = set(OBJECT_NAMES if names is None else names)
    unknown = wanted - set(OBJECT_NAMES)
    if unknown:
        raise ValueError(f"Unknown frozen pilot object(s): {sorted(unknown)}")
    return deepcopy([r for r in FIXED_REQUESTS if r["catalog_id"].split(":", 1)[1] in wanted])


def resolved_selection(names=None, seed=170001):
    """Resolve frozen IDs and check hashes; does not build/compile a simulator."""
    from robocasa_bridge.catalog import resolve_selection
    result = resolve_selection(selection_requests(names), int(seed))
    for row in result:
        relative, expected = FROZEN_MODELS[row["name"]]
        actual = hashlib.sha256((ASSET_ROOT / relative).read_bytes()).hexdigest()
        if row["model_path"] != relative or row["model_sha256"] != expected or actual != expected:
            raise ValueError("Frozen pilot model changed: " + row["name"])
        if min(row["bbox_size_m"][:2]) >= .07:
            raise ValueError("Pilot object has no bbox pinch axis below 7 cm")
    return result


def get_selection(names=None, seed=170001):
    return resolved_selection(names, seed)
