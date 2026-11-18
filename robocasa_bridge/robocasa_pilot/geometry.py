"""Exact collision-mesh containment for RoboCasa pilot geometry scoring v2.

MuJoCo's compiled mesh vertices already include asset scaling. Transform these
vertices with the current geom pose exactly once; never scale them again.
This module contains no expert, control, state write, or model fallback.
"""
from __future__ import annotations

import mujoco
import numpy as np

SCORING_VERSION = "robocasa.pilot.geometry.v2"
TRAY_XY_LIMITS_M = np.array([.143, .108])
BOTTOM_RANGE_M = (-.006, .035)
TOP_LIMIT_M = .27
EEF_CLEARANCE_M = .07
STABLE_SPEED_M_S = .035
LIFT_THRESHOLD_M = .06
STABLE_STEPS = 10


def _id(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError("Missing compiled model element: " + name)
    return int(result)


def collision_bounds(model, data, contact_geoms, tray_body_id):
    """Return collision vertex min/max in the collection tray body frame.

    contact_geoms must be the object's actual contact geom names or IDs.
    Only mesh geoms are supported; unsupported primitives fail explicitly.
    """
    tray_body_id = int(tray_body_id)
    if not 0 <= tray_body_id < model.nbody:
        raise ValueError("Invalid tray body ID")
    center = data.xpos[tray_body_id]
    rotation = data.xmat[tray_body_id].reshape(3, 3)
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    count = 0
    for geom in contact_geoms:
        geom_id = (_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                   if isinstance(geom, str) else int(geom))
        if not 0 <= geom_id < model.ngeom:
            raise ValueError("Invalid contact geom ID")
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            raise NotImplementedError("Exact collision bounds require mesh geom: " + str(name))
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            raise ValueError("Mesh geom has no compiled mesh")
        start = int(model.mesh_vertadr[mesh_id])
        size = int(model.mesh_vertnum[mesh_id])
        if size < 1:
            raise ValueError("Empty compiled collision mesh")
        vertices = np.asarray(model.mesh_vert[start:start + size], dtype=np.float64)
        # geom_xmat includes MuJoCo mesh preprocessing pose; mesh_vert already
        # includes the XML asset scale. Row-vector equivalent of R @ v + p.
        world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        local = (world - center) @ rotation
        if not np.isfinite(local).all():
            raise ValueError("Nonfinite transformed collision geometry")
        lower = np.minimum(lower, local.min(axis=0))
        upper = np.maximum(upper, local.max(axis=0))
        count += 1
    if not count:
        raise ValueError("Object has no contact geoms")
    return lower, upper


def collision_bounds_in_tray(env, name):
    """Return exact object.contact_geoms bounds in the tray frame.

    Accepts a live RoboCasa environment but reads only its compiled model and
    current MuJoCo data. This does not call step, forward, render, or an expert.
    """
    model = getattr(env.sim.model, "_model", env.sim.model)
    data = getattr(env.sim.data, "_data", env.sim.data)
    tray_body_id = env.obj_body_id["collection_tray"]
    return collision_bounds(model, data, env.objects[name].contact_geoms, tray_body_id)


def geometry_conditions(lower, upper, eef_distance):
    """Same containment/clearance thresholds as v1, using physical geometry."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("Collision bounds must be 3-vectors")
    if not np.isfinite(np.r_[lower, upper, eef_distance]).all() or (lower > upper).any():
        raise ValueError("Invalid collision bounds or EEF distance")
    return dict(
        xy=bool((lower[:2] > -TRAY_XY_LIMITS_M).all()
                and (upper[:2] < TRAY_XY_LIMITS_M).all()),
        bottom=bool(BOTTOM_RANGE_M[0] < lower[2] < BOTTOM_RANGE_M[1]),
        upper=bool(upper[2] < TOP_LIMIT_M),
        eef_clear=bool(eef_distance > EEF_CLEARANCE_M),
    )


def stable_placement(conditions, linear_speed, gripper):
    """Physical settling requires open command and unchanged speed threshold."""
    if not np.isfinite(linear_speed) or not np.isfinite(gripper):
        raise ValueError("Nonfinite speed or gripper command")
    return bool(all(conditions.values()) and linear_speed < STABLE_SPEED_M_S and gripper < 0)
