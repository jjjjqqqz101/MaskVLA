"""Read-only pilot data audit; writes only its final report after quotas are met.

Run python -m robocasa_pilot.audit. In-progress collection returns pending without
opening trajectories or writing a premature final report. Historical code hashes
are reported honestly: current files cannot reconstruct missing old source.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import h5py
from .protocol import DATA_ROOT, OUTPUT_ROOT, STATE_DIM, ACTION_DIM, PROMPT, inside_root
from .selection import OBJECT_NAMES, FROZEN_MODELS, ASSET_ROOT
from robocasa_bridge.catalog import asset_path

EXPECTED = {name: {"train": 10, "validation": 2, "test": 2} for name in OBJECT_NAMES}
SPLITS = ("train", "validation", "test")
EXPERT = "scripted_expert_privileged_state"
RAW_SCHEMA = "mask_policy.robocasa.pilot.raw.v1"
HEX = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path):
    digest = hashlib.sha256()
    with inside_root(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _path(value):
    path = inside_root(value)
    if not path.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError("Audit input must be inside pilot_v1 data, not old episodes")
    return path


def _moments():
    return {"n": 0, "state_sum": np.zeros(23), "state_sq": np.zeros(23),
            "action_sum": np.zeros(7), "action_sq": np.zeros(7)}


def _finish_moments(value):
    n = value["n"]
    if not n:
        return {"frames": 0}
    result = {"frames": n}
    for key in ("state", "action"):
        mean = value[key + "_sum"] / n
        std = np.sqrt(np.maximum(value[key + "_sq"] / n - mean ** 2, 0))
        result[key + "_mean"] = mean.tolist()
        result[key + "_std"] = np.maximum(std, 1e-6).tolist()
    return result


def audit(manifest=None, *, full_hashes=True, write=True, progress=None):
    manifest = _path(manifest or DATA_ROOT / "manifest.json")
    if not manifest.is_file():
        return {"status": "pending", "passed": None, "reason": "manifest not yet available", "expected": EXPECTED}
    raw_manifest = manifest.read_bytes()
    payload = json.loads(raw_manifest)
    rows = payload.get("episodes", [])
    counts = {name: {split: 0 for split in SPLITS} for name in OBJECT_NAMES}
    count_errors = []
    for row in rows:
        name, split = row.get("object_id"), row.get("split")
        if name not in counts or split not in SPLITS:
            count_errors.append("Unknown object or split in manifest: " + str(row))
        else:
            counts[name][split] += 1
    over = [f"{name}/{split} exceeds quota" for name in counts for split in SPLITS
            if counts[name][split] > EXPECTED[name][split]]
    complete = counts == EXPECTED and not count_errors
    head = dict(schema="mask_policy.robocasa.pilot.data_audit.v1",
                generated_at=datetime.now(timezone.utc).isoformat(), manifest=str(manifest),
                manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(), expected=EXPECTED,
                counts=counts, split_counts=dict(Counter(x.get("split") for x in rows)),
                episodes=len(rows))
    if not complete:
        return {**head, "status": "failed" if over or count_errors else "pending",
                "passed": False if over or count_errors else None,
                "errors": count_errors + over,
                "reason": "Full HDF5/hash audit runs once all 50 train + 10 validation + 10 test demonstrations are present"}

    errors, warnings, episode_results = [], [], []
    moments = {key: _moments() for key in SPLITS}
    seen_paths, scenes, seeds, rgb_seen = {}, {}, {}, {}
    unique_models, unique_assets, code_versions = {}, {}, defaultdict(set)
    provenance_missing, provenance_recorded = [], []
    raw_hashes = {}
    logical_bytes = 0

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    split_paths = {}
    for split, paths in payload.get("splits", {}).items():
        check(split in SPLITS, "Unknown splits key")
        split_paths[split] = [str(_path(x)) for x in paths]
    for split in SPLITS:
        expected_paths = [str(_path(x["path"])) for x in rows if x["split"] == split]
        check(Counter(split_paths.get(split, [])) == Counter(expected_paths), "splits and episodes disagree: " + split)

    for number, row in enumerate(rows, 1):
        path = _path(row["path"])
        split, name = row["split"], row["object_id"]
        eid = path.parent.name
        result = dict(episode_id=eid, path=str(path), object_id=name, split=split)
        try:
            check(path.name == "trajectory.h5" and path.is_file(), "Missing raw trajectory")
            check(str(path) not in seen_paths, "Episode path duplicated")
            seen_paths[str(path)] = split
            metadata_path, summary_path = path.parent / "metadata.json", path.parent / "summary.json"
            metadata = json.loads(metadata_path.read_text())
            summary = json.loads(summary_path.read_text())
            check(metadata.get("schema") == RAW_SCHEMA, "Wrong raw metadata schema")
            check(metadata.get("source") == EXPERT and summary.get("source") == EXPERT, "Non-expert source")
            check(summary.get("training_ready") is True and summary.get("simulator_success") is True,
                  "Manifest admitted a failed/unapproved demonstration")
            check(summary.get("lift_verified") is True and summary.get("target_lift_max_m", 0) > .06,
                  "No verified physical grasp/lift")
            check(summary.get("stable_placement_steps", 0) >= 10 and summary.get("order_valid") is True,
                  "No verified stable ordered placement")
            check(summary.get("error") is None, "Accepted trajectory has error")
            check(metadata.get("split") == split == summary.get("split"), "Split metadata mismatch")
            check(metadata.get("object_name") == name == summary.get("object_name"), "Object metadata mismatch")
            check(metadata.get("seed") == row.get("seed") == summary.get("seed"), "Seed metadata mismatch")
            check(metadata.get("episode_id") == eid == summary.get("episode_id"), "Episode identity mismatch")
            check(metadata.get("control_freq") == 20, "Control rate differs from 20 Hz")
            check(metadata.get("task") == "single_target_full_pick_place", "Task mismatch")
            check(metadata.get("instruction") == PROMPT, "Target information/prompt contract mismatch")
            check(metadata.get("mask_source") == "sim_gt_offline_annotation", "Mask source mismatch")
            check(metadata.get("policy_input_state_dim") == STATE_DIM and metadata.get("policy_action_dim") == ACTION_DIM,
                  "State/action protocol mismatch")
            contract = metadata["action_contract"]
            check(contract.get("type") == "OSC_POSE" and contract.get("input_type") == "delta"
                  and contract.get("reference_frame") == "world", "Action controller contract mismatch")
            check(contract.get("gripper_open") == -1 and contract.get("gripper_close") == 1,
                  "Gripper sign mismatch")
            check(len(metadata.get("order", [])) == 1, "Pilot trajectory must contain only one target")
            target = metadata["order"][0]
            selected = metadata["selected_objects"]
            check(len(selected) == 1 and selected[0]["name"] == name, "Frozen single-object selection mismatch")
            relative, expected_xml_sha = FROZEN_MODELS[name]
            check(selected[0]["model_path"] == relative and selected[0]["model_sha256"] == expected_xml_sha,
                  "Frozen instance/XML identity changed")
            asset = asset_path(ASSET_ROOT / relative)
            if str(asset) not in unique_assets:
                actual = sha256_file(asset)
                check(actual == expected_xml_sha, "Frozen object XML checksum mismatch")
                unique_assets[str(asset)] = dict(sha256=actual, bytes=asset.stat().st_size)
            compiled = _path(metadata["compiled_model_file"])
            check(compiled.parent == DATA_ROOT / "models", "Compiled model escapes pilot model store")
            if str(compiled) not in unique_models:
                check(compiled.is_file(), "Compiled replay model missing")
                actual = sha256_file(compiled)
                check(actual == metadata["compiled_model_sha256"], "Compiled replay model checksum mismatch")
                unique_models[str(compiled)] = dict(sha256=actual, bytes=compiled.stat().st_size)
            else:
                check(unique_models[str(compiled)]["sha256"] == metadata["compiled_model_sha256"], "Conflicting compiled hash")
            code_hashes = metadata.get("pilot_source_sha256")
            if not code_hashes:
                provenance_missing.append(eid)
            else:
                check(isinstance(code_hashes, dict) and all(HEX.fullmatch(str(v)) for v in code_hashes.values()),
                      "Malformed recorded source SHA256")
                provenance_recorded.append(eid)
                for key, value in code_hashes.items():
                    code_versions[key].add(value)

            scene = metadata.get("scene_family", f"{name}:{metadata['seed']}")
            seed_key = int(metadata["seed"])
            for key, seen, label in ((scene, scenes, "scene_family"), (seed_key, seeds, "seed")):
                check(key not in seen or seen[key] == split, "Cross-split repeated " + label)
                seen[key] = split
            check(row.get("scene_family") == scene, "Scene-family metadata mismatch")
            with h5py.File(path, "r") as f:
                check(_text(f.attrs.get("schema")) == RAW_SCHEMA, "HDF5 schema mismatch")
                check(json.loads(_text(f.attrs["metadata_json"])) == metadata, "HDF5 and sidecar metadata differ")
                obs, act = f["observations"], f["actions"]
                actions = np.asarray(act["arm_gripper"], dtype=np.float64)
                t = len(actions)
                check(actions.shape == (t, 7) and t > 0, "Action shape invalid")
                check(all(len(ds) == t + 1 for ds in obs.values()), "obs[t]/action[t]/obs[t+1] alignment broken")
                check(all(len(ds) == t for ds in act.values()), "Native/policy actions misaligned")
                check(summary["steps"] == row["steps"] == t and summary["observations"] == t + 1,
                      "Summary/manifest length mismatch")
                for key, shape in (("base_rgb", (224,224,3)), ("wrist_rgb", (224,224,3)), ("overview_rgb", (448,448,3))):
                    check(obs[key].shape == (t + 1, *shape) and obs[key].dtype == np.uint8,
                          "Image shape/dtype mismatch: " + key)
                    digest = hashlib.sha256(obs[key][0].tobytes()).hexdigest()
                    check(digest == metadata["initial_rgb_sha256"][key] == row["initial_rgb_sha256"][key],
                          "Initial RGB hash mismatch: " + key)
                    hash_key = (key, digest)
                    check(hash_key not in rgb_seen or rgb_seen[hash_key] == split, "Cross-split identical initial RGB: " + key)
                    rgb_seen[hash_key] = split
                states = np.asarray(obs["robot_state"], dtype=np.float64)
                check(states.shape == (t + 1, 23), "Robot-state shape mismatch")
                for key, ds in obs.items():
                    if key in ("base_rgb", "wrist_rgb", "overview_rgb", "gt_target_mask_offline"):
                        continue
                    if ds.dtype.kind in "fiub":
                        check(np.isfinite(ds[:]).all(), "Non-finite observation: " + key)
                check(np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1.00001, "Action finite/bound check failed")
                check(np.isin(actions[:,6], [-1., 1.]).all(), "Unexpected gripper command")
                native = np.asarray(act["native"], dtype=np.float64)
                check(native.shape == (t,12) and np.isfinite(native).all(), "Native-action contract mismatch")
                check(np.allclose(native[:,:7],actions,atol=1e-6) and np.allclose(native[:,7:11],0)
                      and np.allclose(native[:,11],-1), "Base/torso lock or native-to-policy mapping changed")
                check(np.all(np.diff(obs["frame_id"][:]) == 1), "Frame IDs not consecutive")
                check(np.allclose(np.diff(obs["sim_time"][:]), .05, atol=1e-6), "Simulation time/control rate mismatch")
                targets = [_text(x) for x in obs["target_id"][:]]
                subtasks = obs["subtask_index"][:]
                check(all(x == target for x in targets[:-1]) and np.all(subtasks[:-1] == 0),
                      "An action chunk can cross target contexts")
                check(targets[-1] in (target, ""), "Unexpected terminal target")
                stages = set(_text(x) for x in obs["stage"][:])
                check({"approach","descend","close","lift","transport","place","release","retreat","settle"} <= stages,
                      "Incomplete pick-place action phases")
                masks = obs["gt_target_mask_offline"]
                check(masks.shape == (t + 1,224,224), "Target-mask shape mismatch")
                pixel_counts = []
                for start in range(0,t+1,64):
                    block = masks[start:start+64]
                    check(np.isin(block,[0,1]).all(), "Offline target mask is not binary")
                    pixel_counts.extend(np.count_nonzero(block,axis=(1,2)).tolist())
                check(pixel_counts[0] >= 20, "Initial target visibility too low")
                qc = summary["qc"]
                check(qc.get("alignment") and qc.get("finite") and qc.get("action_bound_ok"), "Recorded QC flags disagree")
                check(qc["initial_target_pixels"] == pixel_counts[0] and qc["min_target_pixels"] == min(pixel_counts)
                      and qc["max_target_pixels"] == max(pixel_counts), "Recorded mask QC disagrees")
                values = moments[split]
                values["n"] += t
                values["state_sum"] += states[:-1].sum(0)
                values["state_sq"] += (states[:-1] ** 2).sum(0)
                values["action_sum"] += actions.sum(0)
                values["action_sq"] += (actions ** 2).sum(0)
                result.update(steps=t, initial_mask_pixels=pixel_counts[0], min_mask_pixels=min(pixel_counts),
                              max_mask_pixels=max(pixel_counts), empty_mask_fraction=float(np.mean(np.asarray(pixel_counts)==0)),
                              action_abs_max=float(np.max(np.abs(actions))), stages=sorted(stages),
                              initial_rgb_sha256=metadata["initial_rgb_sha256"], compiled_model_sha256=metadata["compiled_model_sha256"])
            size = path.stat().st_size
            logical_bytes += size
            if full_hashes:
                raw_hashes[str(path)] = sha256_file(path)
                result["raw_sha256"] = raw_hashes[str(path)]
            result.update(passed=True, raw_bytes=size)
        except Exception as exc:
            message = eid + ": " + str(exc)
            errors.append(message)
            result.update(passed=False,error=str(exc))
        episode_results.append(result)
        if progress and (number % 5 == 0 or number == len(rows)):
            progress(f"audited {number}/{len(rows)} episodes; errors={len(errors)}")

    stats = {key: _finish_moments(value) for key,value in moments.items()}
    training_paths = [str(_path(x["path"])) for x in rows if x["split"] == "train"]
    loader_checks = {"normalization_source_split": "train", "fit_paths": training_paths,
                     "validation_or_test_used_for_fit": False, "no_cross_target_chunks": not errors}
    if not errors:
        from .data import TrajectoryDataset
        train = TrajectoryDataset(payload,"train")
        validation = TrajectoryDataset(payload,"validation")
        try:
            fitted = train.fit_stats()
            check(all(np.allclose(fitted[key], stats["train"][key], atol=1e-5,rtol=1e-4) for key in fitted),
                  "Shared train loader normalization differs from audited TRAIN moments")
            try:
                validation.fit_stats()
            except ValueError:
                loader_checks["validation_fit_rejected"] = True
            else:
                errors.append("Validation dataset allowed normalization fitting")
            loader_checks["train_frames"] = len(train)
            loader_checks["validation_frames"] = len(validation)
        except Exception as exc:
            errors.append("Shared data-loader contract: " + str(exc))
        finally:
            train.close();validation.close()
    if provenance_missing:
        warnings.append(f"{len(provenance_missing)} early episodes lack historical pilot_source_sha256; current hashes cannot reconstruct their source")
    current_source_hashes = {p.name:sha256_file(p) for p in Path(__file__).parent.glob("*.py")}
    if len(code_versions.get("engine.py",set())) > 1:
        warnings.append("Accepted demonstrations record multiple expert source versions; compare action contract, not assumed identical expert code")
    report = {**head, "status": "passed" if not errors else "failed", "passed": not errors,
              "errors": errors, "warnings": warnings, "episode_audits": episode_results,
              "stats_by_split": stats, "normalization": loader_checks,
              "integrity": {"raw_h5_full_sha256_recorded": full_hashes,"raw_h5_sha256":raw_hashes,
                            "object_xml_sha256_verified":unique_assets,"compiled_models_sha256_verified":unique_models,
                            "raw_h5_bytes":logical_bytes,"compiled_model_bytes":sum(x["bytes"] for x in unique_models.values())},
              "source_provenance": {"recorded_episode_count":len(provenance_recorded),
                    "missing_episode_ids":provenance_missing,
                    "historical_sha256_by_filename":{k:sorted(v) for k,v in code_versions.items()},
                    "current_source_sha256":current_source_hashes,
                    "scope":"Recorded historical hashes are provenance records; only matching available current source can be independently checked. Missing historical code is not invented."},
              "verification_scope":"All state/action/mask values, time/context alignment, first RGB SHA, full raw H5 SHA when enabled; image shape and uint8 dtype establish finite pixels. This is data integrity/QC, not a learned-policy success result."}
    if write:
        from .recorder import write_json
        write_json(OUTPUT_ROOT / "data_audit.json",report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",default=str(DATA_ROOT / "manifest.json"))
    parser.add_argument("--skip-payload-hash",action="store_true",help="Quick check; final audit should retain full SHA256")
    parser.add_argument("--no-write",action="store_true")
    args=parser.parse_args()
    report=audit(args.manifest,full_hashes=not args.skip_payload_hash,write=not args.no_write,
                 progress=lambda value:print(value,file=sys.stderr,flush=True))
    brief={key:report[key] for key in ("status","passed","episodes","split_counts","counts","errors","warnings") if key in report}
    if report.get("status") in ("passed","failed") and not args.no_write:
        brief["report"]=str(OUTPUT_ROOT / "data_audit.json")
    print(json.dumps(brief,ensure_ascii=False,indent=2))
    raise SystemExit(1 if report.get("status")=="failed" else 0)


if __name__=="__main__":
    main()
