"""Durable ACT training, bounded recovery, final contract check, service restore."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def last_training_metric(path):
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 131072))
        lines = stream.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if row.get('event') == 'train':
                return row
            if row.get('event') == 'training_start':
                return {}
        except ValueError:
            pass
    return {}


def inspect_checkpoint(path, cfg):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert value['dataset']['fingerprint'] == cfg['dataset_fingerprint']
    assert tuple(value['stream_order']) == ('base_rgb', 'wrist_rgb', 'base_masked_rgb')
    assert value['dataset']['demos_loaded'] == 1250
    assert value['args']['batch_size'] == cfg['batch_size']
    assert value['target_optimizer_steps'] == cfg['total_steps']
    assert value['optimizer_step'] == value['dataloader']['completed_batches']
    return dict(path=str(path), step=value['optimizer_step'], resume_safe=value['resume_safe'])


def gpu_status():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=5)
        used, total, utilization = [int(value.strip()) for value in result.strip().split(',')]
        return dict(used_MiB=used, total_MiB=total, utilization_percent=utilization)
    except Exception:
        return None


parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, type=Path)
args = parser.parse_args()
cfg = json.loads(args.config.read_text())
run = Path(cfg['run_dir'])
source = Path(cfg['source_dir'])
run.mkdir(parents=True, exist_ok=True)
status_path = run / 'job_status.json'
if status_path.exists():
    raise FileExistsError('Existing supervised job preserved; do not launch a duplicate supervisor')
status = dict(status='initializing', optimizer_step=0, target_optimizer_steps=cfg['total_steps'],
              batch_size=cfg['batch_size'], save_every=cfg['save_every'],
              stream_order=['base_rgb', 'wrist_rgb', 'base_masked_rgb'],
              dataset_fingerprint=cfg['dataset_fingerprint'], started_at=time.time(), attempts=[])
child = None
stop_requested = False
training_complete = False


def save_status(**extra):
    status.update(extra, updated_at=time.time())
    atomic_json(status_path, status)


def handle_stop(signum, _frame):
    global stop_requested
    stop_requested = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)


signal.signal(signal.SIGTERM, handle_stop)
signal.signal(signal.SIGINT, handle_stop)
signal.signal(signal.SIGHUP, handle_stop)
save_status()
env = os.environ.copy()
env.update(PYTHONPATH=os.pathsep.join([str(source), cfg['baseline_dir'], cfg['project_dir']]),
           ACT_TRAIN_MEMORY_FRACTION=str(cfg['memory_fraction']),
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='4',
           CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
base_command = [cfg['python'], '-B', '-u', str(source/'training_bootstrap.py'),
    '--data-path', cfg['data_path'], '--run-dir', str(run),
    '--batch-size', str(cfg['batch_size']), '--num-workers', str(cfg['num_workers']),
    '--total-steps', str(cfg['total_steps']), '--save-every', str(cfg['save_every']),
    '--recovery-every', '1000', '--diagnostic-every', '5000', '--log-every', '100',
    '--lr-drop', str((2*cfg['total_steps'])//3), '--amp', 'bf16', '--tf32',
    '--fused-adamw', '--ema-foreach', '--compile-mode', 'default']
resume = None
try:
    for attempt in range(1, 4):
        if stop_requested:
            break
        for relative, expected_hash in cfg['source_sha256'].items():
            assert hashlib.sha256((source/relative).read_bytes()).hexdigest() == expected_hash, relative
        command = base_command + (['--resume', resume] if resume else [])
        row = dict(attempt=attempt, resumed_from=resume, started_at=time.time())
        status['attempts'].append(row)
        log_path = run / f'training_attempt_{attempt}.log'
        with log_path.open('w') as stream:
            child = subprocess.Popen(command, cwd=cfg['baseline_dir'], env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save_status(status='training', trainer_pid=child.pid, current_log=str(log_path))
            stop_deadline = None
            while child.poll() is None:
                if stop_requested:
                    if stop_deadline is None:
                        stop_deadline = time.monotonic() + 300
                    elif time.monotonic() >= stop_deadline:
                        os.killpg(child.pid, signal.SIGKILL)
                metric = last_training_metric(run/'metrics.jsonl')
                step = metric.get('optimizer_step', status['optimizer_step'])
                ips = metric.get('ips', 0)
                eta = (cfg['total_steps']-step)/ips if ips > 0 else None
                save_status(status='stopping' if stop_requested else 'training', optimizer_step=step,
                            latest_metric=metric, estimated_remaining_seconds=eta, gpu=gpu_status())
                time.sleep(10)
            row.update(exit_code=child.returncode, finished_at=time.time())
        child = None
        if stop_requested:
            save_status(status='stopped')
            break
        if row['exit_code'] == 0:
            save_status(status='verifying_final_checkpoint')
            milestones = list(range(cfg['save_every'], cfg['total_steps']+1, cfg['save_every']))
            checked = [inspect_checkpoint(run/'checkpoints'/f'{step}.pt', cfg) for step in milestones]
            assert all(item['step'] == expected and item['resume_safe'] for item, expected in zip(checked, milestones))
            evaluation = [cfg['python'], '-B', str(source/'evaluate_triple_rgb.py'),
                '--checkpoint', str(run/'checkpoints'/f"{cfg['total_steps']}.pt"),
                '--data-path', cfg['data_path'], '--device', 'cpu', '--report', str(run/'final_input_contract.json')]
            cpu_env = dict(env, CUDA_VISIBLE_DEVICES='')
            with (run/'final_contract_check.log').open('w') as stream:
                subprocess.run(evaluation, cwd=source, env=cpu_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            hashes = {}
            for step in milestones:
                path = run/'checkpoints'/f'{step}.pt'
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(4*1024*1024), b''):
                        digest.update(block)
                hashes[path.name] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
            atomic_json(run/'checkpoint_manifest.json', hashes)
            training_complete = True
            save_status(status='training_complete', optimizer_step=cfg['total_steps'],
                        training_complete=True, checkpoint_milestones=milestones)
            break
        candidates = []
        for path in [run/'checkpoints/exception.pt', run/'checkpoints/latest.pt']:
            if path.exists():
                try:
                    item = inspect_checkpoint(path, cfg)
                    if item['resume_safe']:
                        candidates.append(item)
                except Exception as error:
                    row.setdefault('unusable_recovery', []).append(dict(path=str(path), error=str(error)))
        if not candidates or attempt == 3:
            raise RuntimeError('Training failed without a further safe bounded recovery; see attempt logs')
        selected = max(candidates, key=lambda item: item['step'])
        resume = selected['path']
        save_status(status='retrying', optimizer_step=selected['step'], recovery=selected)
        time.sleep(10)
except BaseException as error:
    save_status(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    save_status(training_complete=training_complete, services_restoring=True)
    command = ['python3', cfg['service_manager'], 'resume', '--state-dir', cfg['service_state_dir']]
    with (run/'service_restore.log').open('a') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    service_state = Path(cfg['service_state_dir'])/'service_status.json'
    health = json.loads(service_state.read_text()) if service_state.exists() else {}
    restored = result.returncode == 0 and health.get('status') == 'restored'
    status['services_restoring'] = False
    status['services_restored'] = restored
    status['service_health'] = health
    if training_complete and restored:
        status['status'] = 'completed'
    elif training_complete:
        status['status'] = 'training_complete_restore_needs_attention'
    save_status(finished_at=time.time())
