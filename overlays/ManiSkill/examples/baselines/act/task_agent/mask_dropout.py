"""Bounded task-local mask interruption; no model, image, or simulator imports."""
import json
import math
from pathlib import Path
import time


def _number(value,name,minimum,maximum):
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ValueError(name+' must be a finite number')
    try:value=float(value)
    except OverflowError:raise ValueError(name+' must be a finite number') from None
    if not math.isfinite(value):raise ValueError(name+' must be a finite number')
    if not minimum<=value<=maximum:raise ValueError(name+' is outside the supported range')
    return float(value)


def normalize_mask_dropout(value=None):
    if value is None:value={}
    if not isinstance(value,dict) or set(value)-{'enabled','start_seconds','duration_seconds'}:
        raise ValueError('Invalid mask_dropout configuration')
    enabled=value.get('enabled',False)
    if type(enabled) is not bool:raise ValueError('mask_dropout.enabled must be boolean')
    return dict(enabled=enabled,
        start_seconds=_number(value.get('start_seconds',2),'start_seconds',0,120),
        duration_seconds=_number(value.get('duration_seconds',1),'duration_seconds',.1,30))


class MaskDropout:
    COMMAND_MAX_AGE_SECONDS=10

    def __init__(self,config=None,control_path=None,event_callback=None,clock=None,wall_clock=None):
        self.config=normalize_mask_dropout(config)
        self.control_path=Path(control_path) if control_path else None
        self.emit=event_callback or (lambda event,**values:None)
        self.clock=clock or time.monotonic;self.wall_clock=wall_clock or time.time
        self.automatic_disabled=False
        self.attempt_started=None;self.attempt_generation=None;self.automatic_fired=False
        self.active_until=None;self.source=None;self.last_command_id=None
        self._last_payload=None;self._seen_command_ids=set()
        self.phase='idle';self.generation=0

    def _elapsed(self,now):
        return max(0.,now-self.attempt_started) if self.attempt_started is not None else 0.

    def _end_window(self,reason,now):
        if self.active_until is not None:
            source=self.source;self.active_until=None;self.source=None
            self.emit('mask_dropout_ended',source=source,reason=reason,
                      elapsed_grasp_seconds=self._elapsed(now),step_generation=self.generation)

    def _start_window(self,source,duration,now):
        self._end_window('superseded',now)
        self.source=source;self.active_until=now+duration
        self.emit('mask_dropout_started',source=source,duration_seconds=duration,
                  elapsed_grasp_seconds=self._elapsed(now),step_generation=self.generation,
                  command_id=self.last_command_id if source=='manual' else None)

    def begin_attempt(self,step_generation):
        self.end_attempt('new_attempt')
        self.generation=int(step_generation);self.attempt_generation=self.generation
        self.attempt_started=self.clock();self.automatic_fired=False

    def end_attempt(self,reason='grasp_finished'):
        self._end_window(reason,self.clock())
        self.attempt_started=None;self.attempt_generation=None;self.automatic_fired=False

    def _ignored(self,reason,command_id=None):
        self.emit('mask_dropout_ignored',reason=reason,command_id=command_id,
                  step_generation=self.generation)

    def _read_command(self,now):
        if self.control_path is None:return
        try:
            with self.control_path.open() as stream:payload=stream.read(4097)
        except FileNotFoundError:return
        except OSError as error:
            marker=('read_error',type(error).__name__)
            if marker!=self._last_payload:
                self._last_payload=marker;self._ignored('control_read_error')
            return
        if payload==self._last_payload:return
        self._last_payload=payload
        command_id=None
        try:
            if len(payload)>4096:raise ValueError('Command is too large')
            command=json.loads(payload)
            allowed={'command_id','action','duration_seconds','step_generation','issued_at'}
            if not isinstance(command,dict) or set(command)-allowed:raise ValueError('Invalid command')
            command_id=command.get('command_id')
            if not isinstance(command_id,str) or not command_id or len(command_id)>128:raise ValueError('Invalid command_id')
            if command_id in self._seen_command_ids:return
            self._seen_command_ids.add(command_id);self.last_command_id=command_id
            action=command.get('action')
            if action not in ('trigger','restore'):raise ValueError('Invalid command action')
            generation=command.get('step_generation')
            if type(generation) is not int or generation<0:raise ValueError('Invalid generation')
            issued=_number(command.get('issued_at'),'issued_at',0,1e12)
            if action=='trigger':duration=_number(command.get('duration_seconds'),'duration_seconds',.1,30)
            elif 'duration_seconds' in command:raise ValueError('Restore has no duration')
        except (ValueError,TypeError,json.JSONDecodeError):
            self._ignored('invalid_command',command_id);return
        if action=='restore':
            self._end_window('restore',now);self.automatic_disabled=True
            self.emit('mask_dropout_disabled',reason='restore',command_id=command_id,step_generation=self.generation)
        else:
            age=self.wall_clock()-issued
            if age>self.COMMAND_MAX_AGE_SECONDS:self._ignored('expired',command_id);return
            if age < -2:self._ignored('future_command',command_id);return
            if generation!=self.generation:self._ignored('stale_generation',command_id);return
            if self.phase!='grasp':self._ignored('not_grasp',command_id);return
            if self.attempt_started is None:self._ignored('preflight_not_ready',command_id);return
            self._start_window('manual',duration,now)
        self.emit('mask_dropout_command',action=action,command_id=command_id,step_generation=self.generation)

    def tick(self,phase,step_generation,natural_mask_pixels=0):
        now=self.clock();self.phase=str(phase);self.generation=int(step_generation)
        if self.active_until is not None and now>=self.active_until:self._end_window('elapsed',now)
        if self.attempt_started is not None and (self.phase!='grasp' or self.attempt_generation!=self.generation):
            self.end_attempt('context_changed')
        self._read_command(now)
        elapsed=self._elapsed(now)
        auto_enabled=self.config['enabled'] and not self.automatic_disabled
        if self.phase=='grasp' and self.attempt_started is not None and auto_enabled and not self.automatic_fired and elapsed>=self.config['start_seconds']:
            self.automatic_fired=True
            if self.source=='manual':self._ignored('manual_window_active')
            else:self._start_window('automatic',self.config['duration_seconds'],now)
        active=self.active_until is not None
        return dict(auto_enabled=auto_enabled,disabled=self.automatic_disabled,
            ready=self.attempt_started is not None,active=active,source=self.source if active else None,
            remaining_seconds=max(0.,self.active_until-now) if active else 0.,
            elapsed_grasp_seconds=elapsed if self.attempt_started is not None else None,natural_mask_pixels=int(natural_mask_pixels),
            effective_mask_pixels=0 if active else int(natural_mask_pixels),
            last_command_id=self.last_command_id,step_generation=self.generation)

    def close(self):
        self.end_attempt('close')
        self.emit('mask_dropout_closed',step_generation=self.generation)
