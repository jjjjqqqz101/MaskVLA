"""Bounded, wall-clock visual supervision independent of the control loop.

The configured frequency is the request cadence, not a promise that a remote
VLM completes inference at that frequency. Three concurrent requests can cover
a typical 1--3 second round trip without building an unbounded frame queue.
Only fresh, current-context, monotonically newer responses can drive control.
The caller's judge must use a bounded network timeout; admitted calls cannot be
forcibly interrupted by Python when ``stop(wait=False)`` is requested.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
import copy
import math
import threading
import time
from typing import Callable


def _context(snapshot: dict) -> tuple:
    return (snapshot['step_generation'], snapshot['phase'], snapshot['target_text'])


def _snapshot(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError('Visual supervision requires a snapshot dictionary')
    for name in ('frame_id', 'captured_at', 'phase', 'target_text', 'step_generation'):
        if name not in value:
            raise ValueError('Snapshot missing ' + name)
    if type(value['frame_id']) is not int or value['frame_id'] < 0:
        raise ValueError('Snapshot frame_id must be a nonnegative integer')
    if type(value['step_generation']) is not int or value['step_generation'] < 0:
        raise ValueError('Snapshot step_generation must be a nonnegative integer')
    if not isinstance(value['captured_at'], (int, float)) or not math.isfinite(value['captured_at']):
        raise ValueError('Snapshot captured_at must be a finite monotonic timestamp')
    if not isinstance(value['phase'], str) or not isinstance(value['target_text'], str):
        raise ValueError('Snapshot phase and target_text must be strings')
    # This also isolates image arrays from mutable simulator/camera buffers.
    return copy.deepcopy(value)


class VisualSupervisor:
    """Send latest RGB snapshots to a visual judge on a wall-clock schedule.

    ``judge_callable(snapshot, previous_snapshot)`` returns a validated decision
    dictionary. ``previous_snapshot`` is the preceding dispatched frame from the
    same target/phase/generation, or None. ``on_result(record)`` receives every
    completed request including discarded and errored results, without images.

    ``latest()`` and ``decision()`` return the newest accepted response, or None.
    They re-check age and the current context, so an old grasp confirmation is
    never returned after the backend changes to another target or phase.

    This object is one-shot: create a new supervisor for a new robot session.
    Calling start twice while running is harmless. A stopped supervisor cannot
    be restarted, preventing late responses from leaking into another run.
    """

    def __init__(self, judge_callable: Callable, snapshot_callable: Callable,
                 on_result: Callable | None = None, hz: float = 1.0,
                 max_inflight: int = 3, max_age: float = 4.0):
        if not callable(judge_callable) or not callable(snapshot_callable):
            raise ValueError('judge and snapshot must be callable')
        if on_result is not None and not callable(on_result):
            raise ValueError('on_result must be callable')
        if not isinstance(hz, (int, float)) or not math.isfinite(hz) or not 0 < hz <= 100:
            raise ValueError('hz must be in (0, 100]')
        if type(max_inflight) is not int or not 1 <= max_inflight <= 16:
            raise ValueError('max_inflight must be in 1..16')
        if not isinstance(max_age, (int, float)) or not math.isfinite(max_age) or max_age <= 0:
            raise ValueError('max_age must be positive')
        self.judge = judge_callable
        self.snapshot = snapshot_callable
        self.on_result = on_result
        self.hz = float(hz)
        self.max_inflight = max_inflight
        self.max_age = float(max_age)
        self._period = 1.0 / self.hz
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._running = False
        self._ever_started = False
        self._executor = None
        self._thread = None
        self._prefix = 'visual-judge-' + hex(id(self))[2:]
        self._jobs = {}
        self._next_request = 0
        self._last_dispatched_snapshot = None
        self._last_dispatched_frame = -1
        self._last_accepted_frame = -1
        self._latest = None
        self._records = deque(maxlen=256)
        self._dispatch_times = deque(maxlen=4096)
        self._request_times = deque(maxlen=4096)
        self._result_times = deque(maxlen=4096)
        self._accept_times = deque(maxlen=4096)
        self._latencies = deque(maxlen=4096)
        self._started_at = None
        self._stopped_at = None
        self._scheduler_lag = 0.0
        self._max_scheduler_lag = 0.0
        self._counts = dict(dispatch_count=0, request_count=0, completed_count=0,
                            accepted_count=0, canceled_count=0, error_count=0,
                            snapshot_error_count=0, callback_error_count=0,
                            stale_count=0, context_discarded_count=0,
                            out_of_order_count=0, stopped_discarded_count=0,
                            skipped_no_new_frame=0, skipped_backpressure=0,
                            skipped_deadlines=0)

    def start(self):
        with self._lock:
            if self._running:
                return self
            if self._ever_started:
                raise RuntimeError('Create a new VisualSupervisor after stop')
            self._ever_started = True
            self._running = True
            self._started_at = time.monotonic()
            self._executor = ThreadPoolExecutor(max_workers=self.max_inflight,
                                                thread_name_prefix=self._prefix)
            self._thread = threading.Thread(target=self._schedule,
                                            name=self._prefix+'-scheduler', daemon=True)
            self._thread.start()
        return self

    def stop(self, wait=False):
        """Stop new requests and cancel unstarted jobs; admitted calls may drain.

        The default returns promptly even when remote inference is in flight.
        ``wait=True`` waits for bounded judge calls unless invoked from a judge
        callback itself, where waiting would deadlock the worker thread.
        """
        with self._lock:
            self._running = False
            if self._ever_started and self._stopped_at is None:
                self._stopped_at = time.monotonic()
            self._wake.set()
            thread, executor = self._thread, self._executor
            futures = [job['future'] for job in self._jobs.values()]
        for future in futures:
            future.cancel()
        current = threading.current_thread()
        may_wait = bool(wait) and not current.name.startswith(self._prefix)
        if thread is not None and thread is not current:
            thread.join(timeout=None if may_wait else .05)
        if executor is not None:
            executor.shutdown(wait=may_wait, cancel_futures=True)
        return self.stats()

    def _schedule(self):
        deadline = time.monotonic()
        while True:
            with self._lock:
                if not self._running:
                    return
            delay = deadline-time.monotonic()
            if delay > 0 and self._wake.wait(delay):
                return
            now = time.monotonic()
            with self._lock:
                self._scheduler_lag = max(0.0, now-deadline)
                self._max_scheduler_lag = max(self._max_scheduler_lag, self._scheduler_lag)
            if now-deadline >= self._period:
                missed = int((now-deadline) // self._period)
                with self._lock:
                    self._counts['skipped_deadlines'] += missed
                deadline += missed*self._period
            self._dispatch_tick()
            deadline += self._period

    def _dispatch_tick(self):
        with self._lock:
            if not self._running:
                return
            if len(self._jobs) >= self.max_inflight:
                self._counts['skipped_backpressure'] += 1
                return
        try:
            snapshot = _snapshot(self.snapshot())
        except Exception as error:
            with self._lock:
                self._counts['snapshot_error_count'] += 1
            self._notify(dict(kind='snapshot_error', accepted=False,
                              discard_reason='snapshot_error', completed_at=time.monotonic(),
                              assessment='uncertain', decision='continue',
                              reason='Snapshot unavailable: '+str(error), error=str(error)))
            return
        now = time.monotonic()
        age = now-snapshot['captured_at']
        if age > self.max_age or age < -.5:
            with self._lock:
                self._counts['stale_count'] += 1
            return
        with self._lock:
            if not self._running:
                return
            if snapshot['frame_id'] <= self._last_dispatched_frame:
                self._counts['skipped_no_new_frame'] += 1
                return
            if len(self._jobs) >= self.max_inflight:
                self._counts['skipped_backpressure'] += 1
                return
            previous = self._last_dispatched_snapshot
            if previous is not None and _context(previous) != _context(snapshot):
                previous = None
            self._next_request += 1
            request_id = self._next_request
            job = dict(request_id=request_id, snapshot=snapshot,
                       previous=previous, dispatch_at=now, started_at=None, future=None)
            self._jobs[request_id] = job
            self._last_dispatched_frame = snapshot['frame_id']
            self._last_dispatched_snapshot = snapshot
            self._dispatch_times.append(now)
            self._counts['dispatch_count'] += 1
            # Submitting under this lock makes stop atomic with admission. The
            # worker also checks the running flag before calling the judge.
            future = self._executor.submit(self._invoke, request_id)
            job['future'] = future
        # Register outside the lock: add_done_callback can run synchronously.
        future.add_done_callback(lambda completed: self._completed(request_id, completed))

    def _invoke(self, request_id):
        with self._lock:
            job = self._jobs[request_id]
            if not self._running:
                raise CancelledError()
            job['started_at'] = time.monotonic()
            self._request_times.append(job['started_at'])
            self._counts['request_count'] += 1
            snapshot, previous = job['snapshot'], job['previous']
        # The judge receives its own copies; model wrappers cannot mutate the
        # scheduler's context/freshness tags or the next request's history.
        result = self.judge(copy.deepcopy(snapshot), copy.deepcopy(previous))
        if not isinstance(result, dict):
            raise TypeError('Visual judge must return a decision dictionary')
        return copy.deepcopy(result)

    def _completed(self, request_id, future):
        finished = time.monotonic()
        error = None
        canceled = False
        try:
            result = future.result()
        except CancelledError:
            canceled = True
            result = dict(assessment='uncertain', decision='continue', reason='Request canceled')
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
            result = dict(assessment='uncertain', decision='continue', reason='Visual judge failed', error=error)
        try:
            current = _snapshot(self.snapshot())
            current_error = None
        except Exception as exc:
            current = None
            current_error = str(exc)
        with self._lock:
            job = self._jobs.pop(request_id, None)
            if job is None:
                return
            snap = job['snapshot']
            started = job['started_at']
            latency = finished-started if started is not None else None
            processed_at = time.monotonic()
            age = processed_at-snap['captured_at']
            reason = None
            if canceled:
                self._counts['canceled_count'] += 1
                reason = 'canceled'
            else:
                self._counts['completed_count'] += 1
                self._result_times.append(finished)
                if latency is not None:
                    self._latencies.append(latency)
                if error:
                    self._counts['error_count'] += 1
                    reason = 'judge_error'
                elif not self._running:
                    self._counts['stopped_discarded_count'] += 1
                    reason = 'stopped'
                elif current is None:
                    self._counts['context_discarded_count'] += 1
                    reason = 'current_snapshot_unavailable'
                elif _context(current) != _context(snap):
                    self._counts['context_discarded_count'] += 1
                    reason = 'context_changed'
                elif current['frame_id'] < snap['frame_id']:
                    self._counts['context_discarded_count'] += 1
                    reason = 'frame_rewound'
                elif age > self.max_age or age < -.5:
                    self._counts['stale_count'] += 1
                    reason = 'stale_image'
                elif snap['frame_id'] <= self._last_accepted_frame:
                    self._counts['out_of_order_count'] += 1
                    reason = 'out_of_order'
            record = dict(result, kind='visual_result', judge_result=result,
                          request_id=request_id, frame_id=snap['frame_id'],
                          captured_at=snap['captured_at'], phase=snap['phase'],
                          target_text=snap['target_text'], step_generation=snap['step_generation'],
                          dispatch_at=job['dispatch_at'], started_at=started, completed_at=finished,
                          processed_at=processed_at,
                          frame_age_ms=round(age*1000,3),
                          latency_ms=round(latency*1000,3) if latency is not None else None,
                          queue_latency_ms=round((started-job['dispatch_at'])*1000,3) if started is not None else None,
                          accepted=reason is None, discard_reason=reason)
            if current_error:
                record['current_snapshot_error'] = current_error
            if reason is None:
                self._last_accepted_frame = snap['frame_id']
                self._latest = copy.deepcopy(record)
                self._accept_times.append(finished)
                self._counts['accepted_count'] += 1
            self._records.append(copy.deepcopy(record))
        self._notify(record)

    def _notify(self, record):
        if self.on_result is None:
            return
        try:
            self.on_result(copy.deepcopy(record))
        except Exception:
            with self._lock:
                self._counts['callback_error_count'] += 1

    def latest(self):
        with self._lock:
            if not self._running or self._latest is None:
                return None
            record = copy.deepcopy(self._latest)
        try:
            current = _snapshot(self.snapshot())
        except Exception:
            return None
        if _context(current) != (record['step_generation'], record['phase'], record['target_text']):
            return None
        if time.monotonic()-record['captured_at'] > self.max_age:
            return None
        if current['frame_id'] < record['frame_id']:
            return None
        return record

    def decision(self):
        return self.latest()

    def records(self):
        """Return a bounded recent diagnostic history, without RGB arrays."""
        with self._lock:
            return copy.deepcopy(list(self._records))

    @staticmethod
    def _rate(times):
        return (len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else 0.0

    def stats(self):
        with self._lock:
            latencies = list(self._latencies)
            requests, results = list(self._request_times), list(self._result_times)
            return dict(self._counts, configured_hz=self.hz, requested_hz=self.hz,
                        dispatch_hz=self._rate(list(self._dispatch_times)),
                        request_hz=self._rate(requests), result_hz=self._rate(results),
                        accepted_hz=self._rate(list(self._accept_times)),
                        mean_latency_ms=1000*sum(latencies)/len(latencies) if latencies else None,
                        latency_mean_s=sum(latencies)/len(latencies) if latencies else None,
                        last_latency_ms=1000*latencies[-1] if latencies else None,
                        inflight_count=len(self._jobs), inflight=len(self._jobs), max_inflight=self.max_inflight,
                        scheduler_lag=self._scheduler_lag, max_scheduler_lag=self._max_scheduler_lag,
                        running=self._running, draining=not self._running and bool(self._jobs),
                        started_at=self._started_at, stopped_at=self._stopped_at,
                        last_dispatch_at=self._dispatch_times[-1] if self._dispatch_times else None,
                        last_completed_at=results[-1] if results else None,
                        request_timestamps=requests, completed_timestamps=results,
                        cadence_scope='Wall-clock request admission; remote result latency is measured separately.')
