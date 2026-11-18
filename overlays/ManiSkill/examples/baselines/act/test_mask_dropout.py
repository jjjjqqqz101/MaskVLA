"""Fake-clock contracts for deliberate mask loss; stdlib only."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from task_agent.mask_dropout import MaskDropout,normalize_mask_dropout


class Clock:
    def __init__(self):self.now=100.;self.wall=1000.
    def advance(self,seconds):self.now+=seconds;self.wall+=seconds


class MaskDropoutContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.path=Path(self.tmp.name)/'command.json'
        self.clock=Clock();self.events=[]

    def tearDown(self):self.tmp.cleanup()

    def engine(self,**config):
        return MaskDropout(config,self.path,lambda event,**values:self.events.append((event,values)),
            clock=lambda:self.clock.now,wall_clock=lambda:self.clock.wall)

    def command(self,command_id,action='trigger',generation=3,duration=.5,issued=None):
        command=dict(command_id=command_id,action=action,step_generation=generation,
                     issued_at=self.clock.wall if issued is None else issued)
        if action=='trigger':command['duration_seconds']=duration
        temporary=self.path.with_suffix('.tmp');temporary.write_text(json.dumps(command));temporary.replace(self.path)

    def test_strict_normalization_and_defaults(self):
        self.assertEqual(normalize_mask_dropout(None),dict(enabled=False,start_seconds=2.,duration_seconds=1.))
        self.assertEqual(normalize_mask_dropout({'enabled':True,'start_seconds':0,'duration_seconds':30})['duration_seconds'],30.)
        for value in (False,[],{'enabled':1},{'other':0},{'start_seconds':True},
                      {'start_seconds':float('nan')},{'start_seconds':float('inf')},
                      {'start_seconds':121},{'duration_seconds':False},{'duration_seconds':0},
                      {'duration_seconds':31},{'duration_seconds':'1'}):
            with self.subTest(value=value),self.assertRaises(ValueError):normalize_mask_dropout(value)

    def test_automatic_timer_starts_after_preflight_and_runs_once_per_attempt(self):
        engine=self.engine(enabled=True,start_seconds=2,duration_seconds=1)
        self.clock.advance(50)
        state=engine.tick('grasp',3,10)
        self.assertFalse(state['active']);self.assertFalse(state['ready']);self.assertIsNone(state['elapsed_grasp_seconds'])
        engine.begin_attempt(3);self.clock.advance(1.9)
        self.assertFalse(engine.tick('grasp',3,10)['active'])
        self.clock.advance(.1);state=engine.tick('grasp',3,10)
        self.assertTrue(state['active']);self.assertEqual(state['source'],'automatic')
        self.assertEqual(state['natural_mask_pixels'],10);self.assertEqual(state['effective_mask_pixels'],0)
        self.clock.advance(1.1);self.assertFalse(engine.tick('grasp',3,10)['active'])
        self.clock.advance(20);self.assertFalse(engine.tick('grasp',3,10)['active'])
        engine.end_attempt();engine.begin_attempt(4);self.clock.advance(2)
        self.assertTrue(engine.tick('grasp',4,10)['active'])

    def test_start_zero_activates_first_policy_tick_only_after_begin(self):
        engine=self.engine(enabled=True,start_seconds=0,duration_seconds=.1)
        self.assertFalse(engine.tick('grasp',3,12)['active'])
        engine.begin_attempt(3);self.assertTrue(engine.tick('grasp',3,12)['active'])

    def test_manual_trigger_uses_monotonic_duration(self):
        engine=self.engine();engine.begin_attempt(3)
        self.command('one',duration=1)
        state=engine.tick('grasp',3,5);self.assertTrue(state['active']);self.assertEqual(state['source'],'manual')
        self.clock.wall+=100000
        self.assertTrue(engine.tick('grasp',3,5)['active'])
        self.clock.advance(1.1);self.assertFalse(engine.tick('grasp',3,5)['active'])

    def test_restore_works_old_cross_generation_outside_grasp_and_disables_future_auto(self):
        engine=self.engine(enabled=True,start_seconds=0)
        engine.begin_attempt(3);self.assertTrue(engine.tick('grasp',3,5)['active'])
        self.command('restore',action='restore',generation=0,issued=1)
        state=engine.tick('idle',99,5)
        self.assertFalse(state['active']);self.assertFalse(state['auto_enabled']);self.assertTrue(state['disabled'])
        engine.begin_attempt(100);self.assertFalse(engine.tick('grasp',100,5)['active'])
        self.command('later',generation=100)
        state=engine.tick('grasp',100,5)
        self.assertTrue(state['active']);self.assertEqual(state['source'],'manual')

    def test_invalid_phase_generation_expiry_and_preflight_commands_are_ignored(self):
        cases=[('expired','grasp',3,True,3,980.,'expired'),
               ('oldgen','grasp',3,True,2,1000.,'stale_generation'),
               ('idle','idle',3,False,3,1000.,'not_grasp'),
               ('preflight','grasp',3,False,3,1000.,'preflight_not_ready')]
        for cid,phase,generation,ready,command_generation,issued,reason in cases:
            with self.subTest(reason=reason):
                engine=self.engine()
                if ready:engine.begin_attempt(generation)
                self.command(cid,generation=command_generation,issued=issued)
                self.assertFalse(engine.tick(phase,generation,4)['active'])
                self.assertEqual(self.events[-1][1]['reason'],reason)

    def test_command_replay_and_malformed_payload_are_not_repeated(self):
        engine=self.engine();engine.begin_attempt(3)
        self.command('one');engine.tick('grasp',3,5)
        count=len(self.events);engine.tick('grasp',3,5);self.assertEqual(len(self.events),count)
        self.command('one',duration=2);engine.tick('grasp',3,5);self.assertEqual(len(self.events),count)
        self.path.write_text('{not json');engine.tick('grasp',3,5)
        count=len(self.events);engine.tick('grasp',3,5);self.assertEqual(len(self.events),count)

    def test_context_exit_ends_current_interruption(self):
        engine=self.engine(enabled=True,start_seconds=0);engine.begin_attempt(3)
        self.assertTrue(engine.tick('grasp',3,5)['active'])
        state=engine.tick('transport',3,5)
        self.assertFalse(state['active']);self.assertFalse(state['ready'])
        self.assertEqual(state['effective_mask_pixels'],5)
        self.assertFalse(engine.tick('grasp',3,5)['active'])
        self.assertTrue(any(event=='mask_dropout_ended' and values['reason']=='context_changed' for event,values in self.events))


if __name__=='__main__':unittest.main()
