"""Concurrency/freshness contracts for wall-clock visual supervision."""
import threading
import time
import unittest

from task_agent.supervisor import VisualSupervisor


def wait_until(predicate, timeout=2.0):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.002)
    raise AssertionError('Timed out waiting for asynchronous condition')


class Source:
    def __init__(self, fixed=False):
        self.lock = threading.Lock()
        self.frame = 0
        self.generation = 1
        self.phase = 'grasp'
        self.text = 'the red ball'
        self.fixed = fixed

    def __call__(self):
        with self.lock:
            if not self.fixed:
                self.frame += 1
            return dict(frame_id=self.frame, captured_at=time.monotonic(),
                        phase=self.phase, target_text=self.text,
                        step_generation=self.generation, base_rgb=[[self.frame]],
                        wrist_rgb=[[self.frame]])

    def change(self, phase='transport', generation=2, text='the blue ball'):
        with self.lock:
            self.phase, self.generation, self.text = phase, generation, text


def verdict(**fields):
    return dict(assessment='not_grasped', decision='continue', confidence=.9,
                reason='Visible object remains on table', **fields)


class VisualSupervisionTests(unittest.TestCase):
    def test_wall_clock_dispatch_overlaps_slow_judging_without_blocking_camera(self):
        source = Source()
        lock = threading.Lock()
        active = peak = 0
        def judge(snapshot, previous):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active)
            time.sleep(.085)
            with lock:
                active -= 1
            return verdict()
        supervisor = VisualSupervisor(judge, source, hz=25, max_inflight=3)
        supervisor.start()
        try:
            wait_until(lambda: supervisor.stats()['completed_count'] >= 6)
            stats = supervisor.stats()
            self.assertGreaterEqual(peak,2)
            self.assertLessEqual(peak,3)
            self.assertGreater(stats['request_hz'], 16)
            self.assertLess(stats['request_hz'], 34)
            self.assertGreater(stats['mean_latency_ms'],70)
            self.assertIsNotNone(supervisor.latest())
        finally:
            supervisor.stop(wait=True)

    def test_old_slow_abort_cannot_replace_newer_continue(self):
        source = Source()
        first_started, release_first = threading.Event(), threading.Event()
        lock = threading.Lock()
        count = 0
        def judge(snapshot, previous):
            nonlocal count
            with lock:
                count += 1; sequence=count
            if sequence==1:
                first_started.set();release_first.wait(2)
                return dict(assessment='failed', decision='abort', confidence=.99, reason='old frame')
            return verdict()
        supervisor=VisualSupervisor(judge,source,hz=35,max_inflight=3)
        supervisor.start()
        try:
            self.assertTrue(first_started.wait(1))
            wait_until(lambda: supervisor.stats()['accepted_count']>=1)
            release_first.set()
            wait_until(lambda: any(r['request_id']==1 for r in supervisor.records()))
            record=next(r for r in supervisor.records() if r['request_id']==1)
            self.assertFalse(record['accepted'])
            self.assertEqual(record['discard_reason'],'out_of_order')
            self.assertEqual(supervisor.latest()['decision'],'continue')
        finally:
            release_first.set();supervisor.stop(wait=True)

    def test_target_and_phase_change_discard_inflight_result(self):
        source=Source();started=threading.Event();release=threading.Event()
        def judge(snapshot,previous):
            started.set();release.wait(1)
            return dict(assessment='grasped_lifted',decision='next',confidence=.99,reason='earlier target')
        supervisor=VisualSupervisor(judge,source,hz=2)
        supervisor.start()
        try:
            self.assertTrue(started.wait(1));source.change();release.set()
            wait_until(lambda: len(supervisor.records())>=1)
            record=supervisor.records()[0]
            self.assertFalse(record['accepted'])
            self.assertEqual(record['discard_reason'],'context_changed')
            self.assertIsNone(supervisor.latest())
        finally:
            release.set();supervisor.stop(wait=True)

    def test_latest_rechecks_context_even_before_next_vlm_result(self):
        source=Source();supervisor=VisualSupervisor(lambda *_:verdict(),source,hz=2)
        supervisor.start()
        try:
            wait_until(lambda: supervisor.stats()['accepted_count']==1)
            self.assertIsNotNone(supervisor.latest())
            source.change()
            self.assertIsNone(supervisor.decision())
        finally:
            supervisor.stop(wait=True)

    def test_stale_images_and_expired_latest_never_drive_control(self):
        source=Source()
        def slow(*_):time.sleep(.07);return verdict()
        supervisor=VisualSupervisor(slow,source,hz=2,max_age=.025)
        supervisor.start()
        try:
            wait_until(lambda: supervisor.stats()['completed_count']==1)
            self.assertEqual(supervisor.records()[0]['discard_reason'],'stale_image')
            self.assertIsNone(supervisor.latest())
        finally:supervisor.stop(wait=True)
        supervisor=VisualSupervisor(lambda *_:verdict(),Source(),hz=2,max_age=.04)
        supervisor.start()
        try:
            wait_until(lambda: supervisor.latest() is not None)
            time.sleep(.055)
            self.assertIsNone(supervisor.latest())
        finally:supervisor.stop(wait=True)

    def test_backpressure_does_not_queue_unbounded_requests_and_stop_is_prompt(self):
        release=threading.Event();started=threading.Event();lock=threading.Lock();calls=[]
        def judge(snapshot,previous):
            with lock:calls.append(snapshot['frame_id']);started.set()
            release.wait(2);return verdict()
        supervisor=VisualSupervisor(judge,Source(),hz=70,max_inflight=2)
        supervisor.start()
        try:
            self.assertTrue(started.wait(1))
            wait_until(lambda: supervisor.stats()['request_count']==2)
            wait_until(lambda: supervisor.stats()['skipped_backpressure']>=2)
            self.assertEqual(supervisor.stats()['inflight_count'],2)
            before=time.monotonic();supervisor.stop(wait=False)
            self.assertLess(time.monotonic()-before,.15)
            with lock:admitted=len(calls)
            release.set()
            wait_until(lambda: supervisor.stats()['inflight_count']==0)
            time.sleep(.04)
            with lock:self.assertEqual(len(calls),admitted)
            self.assertIsNone(supervisor.latest())
            self.assertTrue(all(not r['accepted'] for r in supervisor.records()))
        finally:
            release.set();supervisor.stop(wait=True)

    def test_same_frame_is_not_judged_repeatedly(self):
        calls=[]
        supervisor=VisualSupervisor(lambda s,p:(calls.append(s['frame_id']) or verdict()),Source(fixed=True),hz=50)
        supervisor.start()
        try:
            wait_until(lambda: supervisor.stats()['skipped_no_new_frame']>=3)
            self.assertEqual(calls,[0])
        finally:supervisor.stop(wait=True)

    def test_judge_error_is_logged_as_uncertainty_not_success(self):
        events=[]
        def fail(*_):raise RuntimeError('model timeout')
        supervisor=VisualSupervisor(fail,Source(),events.append,hz=2)
        supervisor.start()
        try:
            wait_until(lambda: bool(events))
            self.assertFalse(events[0]['accepted'])
            self.assertEqual(events[0]['assessment'],'uncertain')
            self.assertEqual(events[0]['decision'],'continue')
            self.assertEqual(events[0]['discard_reason'],'judge_error')
            self.assertEqual(supervisor.stats()['error_count'],1)
            self.assertIsNone(supervisor.latest())
        finally:supervisor.stop(wait=True)

    def test_callback_can_read_stats_and_stop_without_deadlock(self):
        done=threading.Event();holder={}
        def callback(record):
            holder['supervisor'].stats()
            holder['supervisor'].stop(wait=True)
            done.set()
        supervisor=VisualSupervisor(lambda *_:verdict(),Source(),callback,hz=10)
        holder['supervisor']=supervisor;supervisor.start()
        try:
            self.assertTrue(done.wait(1))
            self.assertFalse(supervisor.stats()['running'])
            self.assertEqual(supervisor.stats()['callback_error_count'],0)
        finally:supervisor.stop(wait=True)

    def test_judge_cannot_spoof_context_or_mutate_source_buffers(self):
        source=Source();previous=[]
        def judge(snapshot,prior):
            previous.append(prior)
            snapshot['phase']='invented';snapshot['base_rgb'][0][0]=-999
            return verdict(frame_id=-42,phase='invented',step_generation=999,accepted=True)
        supervisor=VisualSupervisor(judge,source,hz=25)
        supervisor.start()
        try:
            wait_until(lambda: supervisor.stats()['accepted_count']>=2)
            record=supervisor.latest()
            self.assertGreater(record['frame_id'],0)
            self.assertEqual(record['phase'],'grasp')
            self.assertEqual(record['step_generation'],1)
            self.assertEqual(previous[1]['phase'],'grasp')
            self.assertGreater(previous[1]['base_rgb'][0][0],0)
        finally:supervisor.stop(wait=True)

    def test_bad_configuration_and_restart_are_explicit(self):
        for settings in ({'hz':0},{'hz':float('nan')},{'max_inflight':0},{'max_age':0}):
            with self.assertRaises(ValueError):VisualSupervisor(lambda *_:verdict(),Source(),**settings)
        supervisor=VisualSupervisor(lambda *_:verdict(),Source())
        supervisor.start();self.assertIs(supervisor.start(),supervisor);supervisor.stop(wait=True)
        with self.assertRaises(RuntimeError):supervisor.start()


if __name__=='__main__':unittest.main()
