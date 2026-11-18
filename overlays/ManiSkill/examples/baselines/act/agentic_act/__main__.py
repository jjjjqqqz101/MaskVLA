"""Run a persistent-scene, multi-step manipulation agent with live Qwen planning."""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

from .contracts import ContractError, Step, TaskSpec, validate_plan, validate_next_action, verify_final_state, StopRequested
from .qwen import QwenClient, QwenPlanner, QwenError


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def make_report(output, summary, events):
    body = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    rows = ''.join('<tr><td>' + html.escape(str(x.get('event'))) + '</td><td><pre>' +
                   html.escape(json.dumps(x, ensure_ascii=False, indent=2)) + '</pre></td></tr>' for x in events)
    video = next(output.glob('*.mp4'), None)
    player = f'<video controls src="{html.escape(video.name)}" style="width:100%;max-width:1100px"></video>' if video else ''
    page = '<!doctype html><meta charset="utf-8"><title>ACT Agent Run</title>' + \
           '<style>body{max-width:1200px;margin:32px auto;font:16px system-ui;background:#f5f7fa;color:#172237}pre{white-space:pre-wrap;word-break:break-word}td{vertical-align:top;padding:12px;border-bottom:1px solid #ccc}img{max-width:48%}</style>' + \
           '<h1>VLM + ACT 多目标顺序放置</h1>' + player + \
           '<p>ACT 抓取与抬升；MPLib 转移、放下与退回。规划和视觉判断使用 Qwen；物理接触与几何检查单独记录。</p>' + \
           '<img src="initial.png"><img src="final.png"><h2>结果</h2><pre>' + body + '</pre><h2>执行记录</h2><table>' + rows + '</table>'
    (output / 'report.html').write_text(page)


def run(args):
    steps = tuple(Step(*item.split(':', 1)) for item in args.order)
    task = TaskSpec(args.instruction, steps)
    output = args.output.resolve()
    if getattr(args, 'web_output', False) and output.is_dir():
        bootstrap = {'web_request.json', 'stdout.log', 'stop.request'}
        entries = list(output.iterdir())
        if any(p.name not in bootstrap or not p.is_file() or p.is_symlink() for p in entries):
            raise ContractError('Web output directory contains existing artifacts; refusing to overwrite.')
    else:
        output.mkdir(parents=True, exist_ok=False)
    events = []
    summary = dict(task=task.as_dict(), success=False, status='initializing',
                   planner='live_qwen', seed=args.seed, attempts={}, completed_steps=0,
                   total_steps=len(task.steps), current_step=0,
                   scope='simulation; simulator instance identity and physical verification assist the agent',
                   skill_backend='existing_ACT_grasp_lift_plus_MPLib_transport_place_retreat',
                   injected_failure=args.inject_first_failure)
    session = None
    client = None
    started = time.monotonic()
    stop_file = getattr(args, 'stop_file', None)

    def check_stop():
        if stop_file and Path(stop_file).exists():
            raise StopRequested('Stop requested; execution is ending cooperatively.')

    def publish(status=None, reason=None, finished=False):
        if status:
            summary['status'] = status
        summary['current_step'] = summary.get('completed_steps', 0)
        snapshot = dict(summary, phase=summary['status'], reason=reason, finished=finished,
                        updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        write_json(output / 'status.json', snapshot)

    def record(event, **details):
        row = dict(event=event, timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), elapsed_seconds=round(time.monotonic()-started, 3), **details)
        events.append(row)
        with (output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    publish('initializing', '准备模型与仿真场景')
    try:
        check_stop()
        client = QwenClient(args.config, output=output)
        planner = QwenPlanner(client)
        planner.step_count = len(task.steps)
        # Import the simulator only after credentials/config pass validation.
        from .backend import RobotSession
        session = RobotSession(checkpoint=args.checkpoint, seed=args.seed, output=output,
                               device=args.device, stop_file=stop_file)
        check_stop()
        image = session.image()
        image.save(output / 'initial.png')
        publish('planning', 'VLM 正在拆解任务与确认执行顺序')
        record('planning_started', total_steps=len(task.steps))
        repair = None
        for _ in range(3):
            check_stop()
            proposed = planner.plan(task.instruction, session.observe(), image, repair)
            check_stop()
            try:
                validated = validate_plan(proposed, task)
                plan = [step.as_dict() for step in validated]
                break
            except ContractError as error:
                repair = str(error)
                record('plan_rejected', reason=repair)
        else:
            raise ContractError('VLM failed to produce the requested ordered plan after three calls.')
        write_json(output / 'plan.json', dict(steps=plan, reason=proposed.get('reason')))
        summary['plan'] = plan
        record('plan_validated', steps=plan)
        publish('deciding', '规划已校验，等待 VLM 选择下一步')
        if args.plan_only:
            summary.update(status='plan_only', plan=plan)
            return summary
        completed = 0
        attempts = {}
        last_result = None
        for _ in range(len(task.steps) * args.max_attempts + 2):
            check_stop()
            state = session.observe()
            # Check earlier placements remain true before launching a later skill.
            physical = verify_final_state(task, state)
            if any(not row['passed'] for row in physical['checks'][:completed]):
                raise RuntimeError('A previously completed placement was disturbed; this minimal agent stops for replanning.')
            publish('deciding', 'VLM 正在根据执行反馈选择下一步')
            repair = None
            for _repair in range(3):
                check_stop()
                decision = planner.next_action(task, plan, state, completed, attempts,
                                               last_result, session.image(), repair)
                check_stop()
                try:
                    decision = validate_next_action(decision, task, completed, attempts, args.max_attempts)
                    break
                except ContractError as error:
                    repair = str(error)
                    record('decision_rejected', reason=repair)
            else:
                raise ContractError('VLM failed to select a valid next action after three calls.')
            record('decision', **decision)
            if decision['decision'] == 'abort':
                summary['status'] = 'planner_aborted'
                break
            if decision['decision'] == 'complete':
                final = verify_final_state(task, session.observe())
                if not final['success']:
                    raise ContractError('Physical final verification rejected completion.')
                summary.update(success=True, status='completed', final_verification=final)
                record('task_completed', verification=final)
                break
            index = completed
            step = task.steps[index]
            attempts[index] = attempts.get(index, 0) + 1
            summary.update(attempts=attempts, completed_steps=completed)
            publish('retrying' if attempts[index] > 1 else 'executing',
                    f'执行第 {index+1}/{len(task.steps)} 步，第 {attempts[index]} 次尝试')
            record('skill_started', step_index=index, attempt=attempts[index],
                   object_id=step.object_id, region_id=step.region_id)
            check_stop()
            before = session.image()
            before.save(output / f'step_{index+1}_attempt_{attempts[index]}_before.png')
            result = session.execute(step.object_id, step.region_id, attempt=attempts[index],
                                     inject_failure=args.inject_first_failure and index == 0 and attempts[index] == 1)
            check_stop()
            state = session.observe()
            physical = verify_final_state(task, state)
            current_ok = physical['checks'][index]['passed']
            after = session.image()
            after.save(output / f'step_{index+1}_attempt_{attempts[index]}_after.png')
            publish('judging', '检查目标位置、释放状态与 VLM 视觉判断')
            record('judging_started', step_index=index, attempt=attempts[index])
            check_stop()
            judgment = planner.judge(step, before, after)
            check_stop()
            record('skill_result', step_index=index, attempt=attempts[index],
                   result=result, physical_check=physical['checks'][index], visual_judgment=judgment)
            # At most one extra view judgment; a visual disagreement never silently becomes success.
            if current_ok and judgment['status'] != 'success':
                judgment = planner.judge(step, before, session.image())
                check_stop()
                record('visual_recheck', step_index=index, judgment=judgment)
            last_result = dict(step_index=index, attempt=attempts[index], skill=result,
                               physical_verified=current_ok, visual_judgment=judgment)
            if current_ok and judgment['status'] == 'success':
                completed += 1
                record('step_completed', step_index=index, object_id=step.object_id, region_id=step.region_id)
            elif current_ok:
                summary['status'] = 'visual_verification_disagreement'
                break
            elif attempts[index] >= args.max_attempts:
                summary['status'] = 'retry_budget_exhausted'
                record('retry_budget_exhausted', step_index=index)
                break
            summary.update(attempts=attempts, completed_steps=completed)
            write_json(output / 'summary.json', summary)
            publish(reason='当前步骤结果已保存')
        else:
            summary['status'] = 'decision_budget_exhausted'
        summary.update(attempts=attempts, completed_steps=completed, plan=plan)
    except StopRequested as error:
        summary.update(success=False, status='stopped')
        record('stopped', reason=str(error))
    except Exception as error:
        message = client.redact(error) if client else str(error)
        summary.update(status='error', error=message)
        record('error', reason=message)
    finally:
        if session:
            try:
                session.image().save(output / 'final.png')
                summary['final_state'] = session.observe()
                summary['final_verification'] = verify_final_state(task, summary['final_state'])
            except Exception as error:
                summary.update(success=False, status='final_verification_error',
                               finalize_error=client.redact(error) if client else str(error))
            finally:
                try:
                    session.close()
                except Exception as error:
                    summary.update(success=False, status='cleanup_error',
                                   cleanup_error=client.redact(error) if client else str(error))
        summary.update(vlm_calls=client.calls if client else 0,
                       elapsed_seconds=round(time.monotonic()-started, 3))
        write_json(output / 'summary.json', summary)
        make_report(output, summary, events)
        publish(finished=True, reason=summary.get('error') or summary['status'])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instruction', default='先把 cube1 放到 zone_A 区域并松手，再把 cube2 放到 zone_B 区域并松手。')
    parser.add_argument('--order', nargs='+', default=['cube1:zone_A','cube2:zone_B'],
                        help='Independent acceptance contract OBJECT:REGION; not given to the initial VLM planner.')
    parser.add_argument('--seed', type=int, default=2026092500)
    parser.add_argument('--max-attempts', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path, help='Private Qwen config outside project.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stop-file', type=Path, help='Cooperative cancellation marker controlled by the web server.')
    parser.add_argument('--web-output', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--inject-first-failure', action='store_true',
                        help='Tag a deliberate no-motion skill timeout to exercise retry; not a naturally observed failure.')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[4] / 'outputs' /
                        ('agentic_act_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(dict(success=result['success'],status=result['status'],output=str(args.output.resolve())),ensure_ascii=False))
    return 0 if result['success'] or args.plan_only and result['status']=='plan_only' else 1


if __name__ == '__main__':
    sys.exit(main())
