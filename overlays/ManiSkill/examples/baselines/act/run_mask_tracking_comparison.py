#!/usr/bin/env python3
"""Run real HTTP mask comparisons on an existing ACT or pi05 simulation lab.

This starts actual full-horizon policy trials. It never fabricates model
responses or provides simulator boxes/masks to the learned perception service.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from summarize_mask_tracking import build_report, summarize_trial, write_report


CONDITIONS = ('oracle', 'perception', 'dropout', 'freeze', 'shift', 'occlusion')


class TrialError(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class LabClient:
    def __init__(self, url, request_timeout=30, poll_interval=1, job_timeout=1200):
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('--url must be an HTTP(S) lab URL without embedded credentials')
        self.url = url.rstrip('/') + '/'
        self.request_timeout, self.poll_interval, self.job_timeout = request_timeout, poll_interval, job_timeout
        self.last_status = None

    def artifact_url(self, path):
        url = urljoin(self.url, path)
        if urlparse(url).netloc != urlparse(self.url).netloc or not urlparse(url).path.startswith('/artifacts/'):
            raise TrialError('Refusing an artifact outside this lab: ' + str(path))
        return url

    def request(self, path, data=None, allow_error=False):
        url = urljoin(self.url, path)
        content = None if data is None else json.dumps(data).encode()
        request = Request(url, data=content, headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            message = error.read(4096).decode(errors='replace')
            raise TrialError(f'HTTP {error.code} from {path}: {message}') from error
        except (URLError, TimeoutError, ValueError) as error:
            raise TrialError(f'Request failed for {path}: {error}') from error
        if not allow_error and isinstance(result, dict) and result.get('error'):
            raise TrialError(str(result['error']))
        return result

    def status(self):
        status = self.request('/api/status', allow_error=True)
        if not isinstance(status, dict) or type(status.get('busy')) is not bool:
            raise TrialError('Invalid lab status schema')
        self.last_status = status
        return status

    def wait_idle(self, operation):
        deadline = time.monotonic() + self.job_timeout
        while True:
            status = self.status()
            if status.get('error'):
                raise TrialError(operation + ': ' + str(status['error']))
            if not status['busy']:
                return status
            if time.monotonic() >= deadline:
                raise TrialError(operation + f' exceeded {self.job_timeout:g} seconds')
            time.sleep(self.poll_interval)

    def submit(self, operation, data):
        self.request('/api/' + operation, data)
        return self.wait_idle(operation)

    def download(self, path, destination):
        url = self.artifact_url(path)
        try:
            with urlopen(url, timeout=self.request_timeout) as response, destination.open('wb') as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            destination.unlink(missing_ok=True)
            raise TrialError('Artifact download failed: ' + path + ': ' + str(error)) from error
        return url


def canonical_checkpoint(client, policy, requested):
    entries = client.request('/api/checkpoints')
    if not isinstance(entries, list):
        raise TrialError('Invalid checkpoint list schema')
    for entry in entries:
        path = entry.get('path', '')
        if policy == 'pi05' and requested in (entry.get('name'), path):
            return entry['name'], path
        if policy == 'act' and (requested == path or requested.endswith('/' + path)):
            return path, path
    raise TrialError('Requested checkpoint is not available in this lab: ' + requested)


def mask_config(args, condition):
    config = dict(mask_source='oracle' if condition == 'oracle' else 'perception',
                  target_text=args.target_text, loss_mode=args.loss_mode,
                  intervention=dict(kind='none' if condition in ('oracle', 'perception') else condition,
                                    start=args.fault_start, duration=args.fault_duration,
                                    dx=args.shift_dx, dy=args.shift_dy))
    config['endpoint' if args.policy == 'act' else 'perception_url'] = args.perception_url
    return config


def obtain_result(client, status, previous_ids, scene_id, target, checkpoint_suffix):
    records = [row for row in status.get('results', []) if row.get('id') not in previous_ids
               and row.get('scene_id') == scene_id and row.get('target') == target]
    if len(records) != 1:
        raise TrialError(f'Expected exactly one new result for this scene/target, got {len(records)}')
    record = records[0]
    if not str(record.get('checkpoint', '')).endswith(checkpoint_suffix):
        raise TrialError('Returned result used a different checkpoint')
    if record.get('error') or record.get('cancelled') or record.get('stop_reason') in ('perception_error', 'inference_error', 'exception', 'error', 'cancelled'):
        raise TrialError('Trial stopped with an operational error: ' + str(record.get('error') or record.get('stop_reason')))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True, choices=('act', 'pi05'))
    parser.add_argument('--url', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--target', required=True, help='ACT one-based target number; pi05 object name, e.g. 011_banana')
    parser.add_argument('--target-text', required=True, help='Visible natural-language description, sent to the VLM without simulator labels')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=['oracle', 'perception', 'dropout'])
    parser.add_argument('--loss-mode', choices=('observe', 'pause', 'reacquire'), default='observe')
    parser.add_argument('--perception-url', default='http://127.0.0.1:18830')
    parser.add_argument('--fault-start', type=int, default=50)
    parser.add_argument('--fault-duration', type=int, default=30)
    parser.add_argument('--shift-dx', type=int, default=20)
    parser.add_argument('--shift-dy', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--allow-heldout', action='store_true')
    parser.add_argument('--poll-interval', type=float, default=1)
    parser.add_argument('--timeout', type=float, default=1200, help='Maximum seconds for each scene or full policy trial')
    parser.add_argument('--request-timeout', type=float, default=30)
    parser.add_argument('--download-media', action='store_true', help='Also download each video and initial/final PNG')
    parser.add_argument('--output', required=True, type=Path, help='New or empty output directory on the machine running this script')
    args = parser.parse_args()
    if not 0 <= args.seed < 2**31 or not 0 <= args.fault_start <= 500 or not 1 <= args.fault_duration <= 500:
        parser.error('seed or fault interval is outside the backend limits')
    if not args.target_text.strip() or len(set(args.conditions)) != len(args.conditions):
        parser.error('Provide target text and unique conditions')
    if not 0 < args.poll_interval <= 60 or min(args.timeout, args.request_timeout) <= 0:
        parser.error('Invalid timeout or polling interval')
    try:
        target = int(args.target) if args.policy == 'act' else args.target
        if args.policy == 'act' and target < 1:
            raise ValueError('ACT target must be a positive one-based index')
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Output directory must be new or empty; existing records are preserved')
        args.output.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    manifest = dict(schema_version=1, status='running', started_utc=utc(), policy=args.policy,
                    seed=args.seed, target=target, checkpoint_requested=args.checkpoint,
                    target_text=args.target_text, conditions=args.conditions, loss_mode=args.loss_mode,
                    url=args.url, runs=[], fair_comparison_scope='Within this policy/environment only')
    save(args.output/'comparison.json', manifest)
    client, phase, summaries = None, 'initialization', []
    active_job = False
    try:
        client = LabClient(args.url, args.request_timeout, args.poll_interval, args.timeout)
        initial = client.status()
        if initial.get('error'):
            raise TrialError('Lab reports an existing error: ' + str(initial['error']))
        if initial['busy']:
            raise TrialError('Lab is busy; refusing to interfere with an existing run')
        if initial.get('perception_supported') is not True:
            raise TrialError('This lab does not expose the learned-mask API')
        checkpoint, checkpoint_suffix = canonical_checkpoint(client, args.policy, args.checkpoint)
        manifest['checkpoint_api_name'] = checkpoint
        scene_id = None
        for condition in args.conditions:
            phase = condition + ':scene'
            config = mask_config(args, condition)
            print(utc() + ' START ' + args.policy + ' ' + condition, flush=True)
            if args.policy == 'pi05' or scene_id is None:
                body = dict(seed=args.seed, target=target)
                if args.policy == 'pi05':
                    body.update(allow_heldout=args.allow_heldout, mask_config=config)
                active_job = True
                status = client.submit('scene', body)
                active_job = False
                scene = status.get('scene')
                if not isinstance(scene, dict) or not scene.get('id') or scene.get('seed') != args.seed:
                    raise TrialError('Scene generation returned an unexpected seed or schema')
                scene_id = scene['id']
            else:
                status = client.status()
                if status.get('scene', {}).get('id') != scene_id:
                    raise TrialError('ACT scene changed between conditions')
            previous_ids = {row.get('id') for row in status.get('results', [])}
            field = 'perception' if args.policy == 'act' else 'mask_config'
            body = dict(scene_id=scene_id, target=target, checkpoint=checkpoint, **{field: config})
            if args.policy == 'pi05':
                body['gpu'] = args.gpu
            phase = condition + ':run'
            active_job = True
            status = client.submit('run', body)
            active_job = False
            record = obtain_result(client, status, previous_ids, scene_id, target, checkpoint_suffix)
            if record.get('seed') != args.seed:
                raise TrialError('Trial result seed does not match the requested seed')
            folder = args.output/condition
            folder.mkdir()
            phase = condition + ':artifacts'
            metrics_path = record.get('metrics')
            if not isinstance(metrics_path, str) or not metrics_path.endswith('/result.json'):
                raise TrialError('Trial did not expose its result artifact URL')
            result_url = client.download(metrics_path, folder/'result.json')
            trace_path = metrics_path.rsplit('/', 1)[0] + '/step_metrics.json'
            trace_url = client.download(trace_path, folder/'step_metrics.json')
            result = json.loads((folder/'result.json').read_text())
            if result.get('id') != record.get('id'):
                raise TrialError('Downloaded result does not match the status record')
            traces = json.loads((folder/'step_metrics.json').read_text())
            summary = summarize_trial(result, traces, args.policy, condition, folder/'result.json')
            summaries.append(summary)
            row = dict(condition=condition, scene_id=scene_id, trial_id=result['id'],
                       request=body, result_path=str(folder/'result.json'),
                       trace_path=str(folder/'step_metrics.json'), result_url=result_url, trace_url=trace_url)
            if args.download_media:
                for key, name in [('video', 'video.mp4'), ('initial', 'initial.png'), ('final', 'final.png')]:
                    row[key + '_url'] = client.download(result[key], folder/name)
            manifest['runs'].append(row)
            report = build_report(summaries)
            write_report(report, args.output)
            save(args.output/'comparison.json', manifest)
            if summary['operational_error']:
                raise TrialError('Trial artifact contains an operational error: ' + str(summary['operational_error']))
            if not report['all_initial_inputs_match']:
                raise TrialError('Initial RGB/state hashes are missing or differ; comparison is not reproducible')
            print(utc() + f" FINISH {condition} grasp_success={summary['actual_success']} stop={summary['stop_reason']}", flush=True)
        manifest.update(status='complete', finished_utc=utc(), initial_inputs_match=True)
        save(args.output/'comparison.json', manifest)
        print(json.dumps(dict(status='complete', trials=len(summaries), report=str(args.output/'summary.json'),
                              note='Task failures remain measured outcomes; this status only means all requested experiments completed without an operational error.'), ensure_ascii=False), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status='failed', failed_phase=phase, error=type(error).__name__ + ': ' + str(error), finished_utc=utc())
        if active_job and client is not None:
            try:
                client.request('/api/stop', {})
                manifest['stop_requested'] = True
            except Exception as stop_error:
                manifest['stop_request_error'] = str(stop_error)
        if client is not None:
            manifest['last_status'] = client.last_status
        save(args.output/'comparison.json', manifest)
        if summaries:
            write_report(build_report(summaries), args.output)
        print('Comparison failed at ' + phase + ': ' + manifest['error'], file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == '__main__':
    sys.exit(main())
