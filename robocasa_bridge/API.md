# RoboCasa bridge HTTP contract v2 (backward compatible)

Loopback service: 127.0.0.1:18850. Existing web console proxies /api/robocasa/<path>.
GET /health, /config, /status, /frame.png, /episodes.

## Object catalog and scene selection

GET /catalog?query=apple&source=objaverse&offset=0&limit=30&refresh=1
returns categories, total, offset, limit, sources, downloaded_instances.
Each category contains catalog_id (e.g. objaverse:apple), name, name_zh,
source, types, instance_count, available, selectable, graspable, disabled_reason.
Source IDs are objaverse, aigen, lightwheel (aigen maps to disk aigen_objs).
GET /catalog?category_id=objaverse:apple&offset=0&limit=50 adds instances:
instance_id, label, selectable, disabled_reason, bbox_size_m, original_bbox_size_m,
applied_scale_factor and scale. Instance IDs are stable hashes of canonical
asset-relative paths. No request can supply a filesystem path.
Aigen has one exact legacy author texture path; a shared resolver maps only that
path to ASSET/shared/textures/ceramic.png, pinned to the official texture SHA256.
All other outside references are rejected. Source XML remains unchanged and
selected_objects.asset_reference_relocations records the mapping/provenance.

POST /scene {"seed":17,"objects":[
  {"catalog_id":"objaverse:apple","instance_id":null,"quantity":1},
  {"catalog_id":"objaverse:banana","quantity":1}
]}
The expanded list must contain 1–6 objects; its order defines subtask order.
Omitted/null instance_id means a seed-deterministic suitable variant.
Each official model uses upstream category physics and MJCFObject, with a
recorded uniform fit factor where needed. Max fitted bbox is 0.13×0.13×0.18m;
fitting below 0.45 of official category scale is rejected. Official non-graspable
decor, auxiliary parts, missing resources, unsafe paths, oversized models and
overlapping/out-of-counter placements are rejected clearly.
Category graspable labels do NOT assert that a learned policy or generic
scripted expert can grasp the model.
The existing green tray is retained. Generic scenes support manual collection
and both replay modes, but no scripted/learned/VLM execution is claimed.

Default backward-compatible scene:
POST /scene {"seed":17,"order":["red_cube","blue_cube"]}
or {"preset":"classic2","seed":17}. The default red/blue expert remains available.
POST /scene {"restore":true} restores the current initial physical layout.
POST /scene {"reuse_episode":"rc_..."} restores the concrete recorded models,
scales, full state and exact compiled MJB; it does not resample the category.
Scene status selected_objects and raw metadata preserve runtime id, category,
registry, concrete instance, model-relative path/XML SHA256, scale and initial
poses. Legacy episodes without selected_objects retain the classic behavior.

## Manual controls, recording and targets

POST /record/start {"label":"manual"}
POST /control {"translation":[0.01,0,0],"rotation":[0,0,0],"gripper":-1,"steps":1}
Translation is world-frame meters PER step; rotation is world-frame axis-angle
radians PER step; -1 opens and +1 closes the gripper. Base/torso stay zero.

POST /target {"target_id":"item_02_banana"} fixes the offline annotation target.
POST /target {"target_id":null} returns to completion-driven automatic target.
This only changes target_id/subtask context and the single-target GT annotation;
it does not step physics, move objects or mark a task complete.
It is allowed while recording and splits training chunks at the target boundary.
status.target_override is null or the runtime target id.
progress.total and current_target use the actual selected target list.

POST /record/stop {"quality":"success|failure|stopped","note":"..."}
POST /scripted_demo {} restores the default cube scene and runs real contact
controls. Nonclassic scenes explicitly reject it. Source is
scripted_expert_privileged_state; success is never assumed.
POST /stop {} cooperatively stops jobs and retains recordings, even on failure.

## Replay and artifacts

POST /replay {"episode_id":"rc_...","mode":"states|actions"}
GET /artifacts/<episode_id>/<filename> supports bounded public PNG/MP4/JSON/H5/NPZ
downloads and byte ranges. Episode list includes exact selected_objects.
All mutations are asynchronous: {ok:true,queued:true}; inspect /status.

State replay restores recorded states. Action replay executes saved native
controls and reports trajectory divergence separately. Recorded target context is
restored for each replay frame. Raw schema remains v1: observations T+1 and
actions T; dynamic qpos/integration sizes are saved and compiled model is pinned.
The 23D robot state / 7D arm-gripper policy mapping stays unchanged.
GT masks remain sim_gt_offline_annotation, never SAM2/Cutie. ACT dual/triple and
OpenPI sample adapters preserve stream order and stop at subtask boundaries.
The separate five-object pilot has its own RoboCasa-specific training and inference
entry points in robocasa_pilot/. Supply a compatible policy checkpoint to run them.
