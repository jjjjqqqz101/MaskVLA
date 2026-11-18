#!/usr/bin/env python3
"""Summarize actual ACT/pi05 trial artifacts without running either policy."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys


LOST_STATES = {'lost', 'reacquiring', 'failed'}
OPERATIONAL_STOPS = {'perception_error', 'inference_error', 'exception', 'error', 'cancelled'}


def finite(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric metric, got ' + repr(value))
    return float(value)


def first(mapping, *keys):
    return next((mapping[key] for key in keys if mapping.get(key) is not None), None)


def infer_policy(result):
    if 'mask_config' in result or 'initial_base_sha256' in result:
        return 'pi05'
    if 'perception_config' in result or str(result.get('checkpoint', '')).endswith('.pt'):
        return 'act'
    raise ValueError('Cannot infer policy from this result; pass --policy')


def summarize_trial(result, traces, policy=None, condition=None, result_path=None):
    if not isinstance(result, dict) or not isinstance(result.get('id'), str):
        raise ValueError('Expected an actual backend result object with an id')
    if not isinstance(traces, list) or any(not isinstance(row, dict) for row in traces):
        raise ValueError('step_metrics.json must be an array of frame objects')
    policy = policy or infer_policy(result)
    config = result.get('perception_config', result.get('mask_config', {}))
    intervention = config.get('intervention', {})
    kind = intervention.get('kind', 'none')
    source = result.get('mask_source', config.get('mask_source'))
    oracle = source in ('oracle', 'simulator_instance_segmentation')
    condition = condition or ('oracle' if oracle else 'perception' if kind == 'none' else kind)
    rows = [row.get('control_perception', row.get('perception', {})) for row in traces]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Per-frame perception metadata must be objects')
    ious, latencies, qualities = [], [], []
    empty = lost = natural_lost = intervention_frames = policy_input_lost = paused = consumed_steps = 0
    generations, events, seen_events = set(), [], set()
    for trace, row in zip(traces, rows):
        iou = finite(first(row, 'gt_iou', 'mask_iou'))
        if iou is not None:
            if not 0 <= iou <= 1:
                raise ValueError('Mask IoU is outside [0, 1]')
            ious.append(iou)
        latency = finite(row.get('latency_ms'))
        if latency is not None and latency > 0:
            latencies.append(latency)
        quality = finite(row.get('quality'))
        if quality is not None:
            qualities.append(quality)
        pixels = finite(first(row, 'policy_mask_pixels', 'perceived_mask_pixels', 'mask_pixels'))
        empty += int(pixels == 0)
        lost += int(row.get('state') in LOST_STATES)
        natural_lost += int(row.get('natural_lost') is True)
        intervention_frames += int(row.get('intervention_active') is True)
        is_paused = bool(trace.get('paused_for_mask', trace.get('paused', False)))
        paused += int(is_paused)
        consumed = trace.get('policy_input_perception')
        if ('policy_input_perception' not in trace and policy == 'act' and not is_paused
                and result.get('inference', {}).get('temporal_agg') is True):
            consumed = row
        if isinstance(consumed, dict):
            consumed_steps += 1
            policy_input_lost += int(consumed.get('state') in LOST_STATES)
        if isinstance(row.get('generation'), int):
            generations.add(row['generation'])
        for event in row.get('events', []):
            if not isinstance(event, dict):
                raise ValueError('Perception event must be an object')
            key = json.dumps(event, sort_keys=True, ensure_ascii=False)
            if key not in seen_events:
                seen_events.add(key)
                events.append(event)
    attempts = max([int(row.get('reacquires', 0)) for row in rows] + [0])
    successes = sum(event.get('type') == 'reacquired' for event in events)
    operational_error = result.get('error') or (result.get('stop_reason') if result.get('stop_reason') in OPERATIONAL_STOPS else None)
    operational_error = operational_error or ('cancelled' if result.get('cancelled') else None)
    actual_success = result.get('correct_pick')
    if not isinstance(actual_success, bool):
        actual_success = bool(result.get('success_once')) and not bool(result.get('wrong_object_grasped_ever'))
    if operational_error:
        actual_success = False
    hashes = {key: value for key, value in result.items()
              if key.startswith('initial_') and key.endswith('_sha256')}
    return dict(policy=policy, condition=condition, trial_id=result['id'], seed=result.get('seed'),
                target=result.get('target'), checkpoint=result.get('checkpoint'),
                checkpoint_sha256=result.get('checkpoint_sha256'), mask_source=source,
                loss_mode=config.get('loss_mode', 'observe'), intervention=intervention,
                result_path=str(result_path) if result_path else None,
                frames=len(traces), steps=result.get('steps'), lost_frames=lost,
                natural_lost_frames=natural_lost, empty_mask_frames=empty,
                intervention_active_frames=intervention_frames,
                mean_gt_iou=statistics.fmean(ious) if ious else None,
                minimum_gt_iou=min(ious) if ious else None,
                median_perception_ms=statistics.median(latencies) if latencies else None,
                mean_perception_ms=statistics.fmean(latencies) if latencies else None,
                latency_samples=len(latencies),
                latency_scope='Per-frame service processing; excludes policy inference and simulation. Zero oracle latencies omitted.',
                median_uncalibrated_quality=statistics.median(qualities) if qualities else None,
                paused_steps=paused,
                policy_input_lost_steps=policy_input_lost if consumed_steps else None,
                policy_input_metadata_steps=consumed_steps,
                reacquire_attempts=attempts, reacquire_successes=successes,
                tracker_generations=sorted(generations),
                actual_success=actual_success, correct_pick=result.get('correct_pick'),
                success_once=result.get('success_once'), success_at_end=result.get('success_at_end'),
                wrong_object_grasped_ever=result.get('wrong_object_grasped_ever'),
                complete=result.get('complete'), stop_reason=result.get('stop_reason'),
                operational_error=operational_error, hashes=hashes,
                video=result.get('video'), metrics=result.get('metrics'))


def build_report(trials):
    groups = {}
    for row in trials:
        key = (row['policy'], row.get('checkpoint'), row.get('seed'), str(row.get('target')))
        groups.setdefault(key, []).append(row)
    comparisons = []
    for (policy, checkpoint, seed, target), rows in groups.items():
        keys = ['initial_rgb_sha256', 'initial_state_sha256'] if policy == 'act' else [
            'initial_base_sha256', 'initial_wrist_sha256', 'initial_state_sha256']
        checks = {}
        for key in keys:
            values = [row.get('hashes', {}).get(key) for row in rows]
            present = all(isinstance(value, str) and len(value) == 64
                          and set(value.lower()) <= set('0123456789abcdef') for value in values)
            checks[key] = dict(present_for_all=present, equal=present and len(set(values)) == 1,
                               values=dict(zip((row['condition'] + ':' + row['trial_id'] for row in rows), values)))
        valid = all(check['equal'] for check in checks.values())
        comparisons.append(dict(policy=policy, checkpoint=checkpoint, seed=seed, target=target,
                                conditions=[row['condition'] for row in rows], trials=len(rows),
                                initial_inputs_match=valid, checks=checks,
                                comparison_scope='Within this policy/environment only; ACT and pi05 environments are not a fair model ranking.'))
    return dict(schema_version=1, trials=trials, comparisons=comparisons,
                all_initial_inputs_match=bool(comparisons) and all(row['initial_inputs_match'] for row in comparisons),
                operational_errors=[dict(trial_id=row['trial_id'], error=row['operational_error'])
                                    for row in trials if row.get('operational_error')],
                interpretation='Single-seed outcomes are observations, not success-rate estimates. GT IoU is evaluation-only; tracker quality is not calibrated correctness.')


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    fields = ['policy', 'condition', 'seed', 'target', 'trial_id', 'frames', 'lost_frames',
              'natural_lost_frames', 'empty_mask_frames', 'mean_gt_iou', 'median_perception_ms',
              'paused_steps', 'reacquire_attempts', 'reacquire_successes', 'actual_success',
              'wrong_object_grasped_ever', 'stop_reason', 'operational_error']
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['trials'])
    lines = ['Mask tracking comparison', '',
             'Each comparison uses the same policy, checkpoint, seed and target. ACT and π0.5 use different environments and are not ranked against one another.', '',
             '| Policy | Condition | Lost frames | Mean GT IoU | Median mask ms | Paused steps | Reacquire attempts/successes | Actual grasp success | Stop reason |',
             '|---|---|---:|---:|---:|---:|---:|---|---|']
    def cell(value):
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    for row in report['trials']:
        vals = [row['policy'], row['condition'], row['lost_frames'], row['mean_gt_iou'],
                row['median_perception_ms'], row['paused_steps'],
                f"{row['reacquire_attempts']}/{row['reacquire_successes']}", row['actual_success'], row['stop_reason']]
        lines.append('| ' + ' | '.join(map(cell, vals)) + ' |')
    lines += ['', 'Initial RGB/state hashes match within all comparison groups: ' + str(report['all_initial_inputs_match']),
              '', report['interpretation']]
    if report['operational_errors']:
        lines += ['', 'Operational errors: ' + json.dumps(report['operational_errors'], ensure_ascii=False)]
    (output/'summary.md').write_text('\n'.join(lines) + '\n')


def load_trials(paths, policy=None):
    records, seen = [], set()
    for path in paths:
        path = Path(path)
        candidates = sorted(path.rglob('result.json')) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.resolve() in seen:
                continue
            seen.add(candidate.resolve())
            result = json.loads(candidate.read_text())
            traces_path = candidate.with_name('step_metrics.json')
            if not traces_path.is_file():
                raise FileNotFoundError('Missing actual trace artifact: ' + str(traces_path))
            traces = json.loads(traces_path.read_text())
            condition = candidate.parent.name if candidate.parent.name in ('oracle', 'perception', 'dropout', 'freeze', 'shift', 'occlusion') else None
            records.append(summarize_trial(result, traces, policy, condition, candidate))
    if not records:
        raise ValueError('No result.json artifacts found')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='*', type=Path)
    parser.add_argument('--input', action='append', default=[], type=Path)
    parser.add_argument('--policy', choices=('act', 'pi05'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        paths = args.inputs + args.input
        if not paths:
            parser.error('Provide actual trial directories or result.json files')
        report = build_report(load_trials(paths, args.policy))
        write_report(report, args.output)
        print(json.dumps(dict(trials=len(report['trials']), initial_inputs_match=report['all_initial_inputs_match'],
                              operational_errors=report['operational_errors'], report=str(args.output/'summary.json')), ensure_ascii=False))
        return 0 if report['all_initial_inputs_match'] and not report['operational_errors'] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('Summary failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
