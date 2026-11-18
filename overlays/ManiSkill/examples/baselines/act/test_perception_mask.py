"""Behavior tests for RGB-only perception, controlled loss, recovery and transport."""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from perception_mask.client import PerceptionMaskClient, encode_rgb
from perception_mask.core import TrackingSession, validate_box, shift_mask
from perception_mask.server import Application, Handler


class Locator:
    def __init__(self):self.calls=[]
    def locate(self,rgb,text):
        self.calls.append((rgb.copy(),text))
        return dict(found=True,bbox_xyxy=[6,6,14,14],coordinate_space='pixels',confidence=.9)


class Segmenter:
    checkpoint=Path('/test/sam2')
    def segment(self,rgb,box):
        mask=np.zeros(rgb.shape[:2],bool);mask[6:14,6:14]=True
        return mask,.95


class Tracker:
    def __init__(self,results=None):
        self.results=list(results or []);self.rejected=0;self.accepted=0;self.closed=False
    def initialize(self,rgb,mask):self.mask=mask.copy()
    def predict(self,rgb):return self.results.pop(0) if self.results else (self.mask.copy(),.95)
    def accept(self):self.accepted+=1
    def reject(self):self.rejected+=1
    def close(self):self.closed=True


class Factory:
    checkpoint=Path('/test/cutie')
    def __init__(self,results=None):self.instances=[];self.results=results
    def __call__(self):
        tracker=Tracker(self.results if not self.instances else None)
        self.instances.append(tracker);return tracker


class TrackingBehavior(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((32,32,3),160,np.uint8);self.locator=Locator();self.factory=Factory()
    def session(self,options=None,factory=None):
        return TrackingSession(self.locator,Segmenter(),factory or self.factory,'left red cube',options)

    def test_box_coordinates_are_explicit_and_strict(self):
        self.assertEqual(validate_box([100,200,500,600],200,100,'normalized_1000'),[20,20,100,60])
        for box in ([4,3,2,6],[-1,0,9,9],[0,0,float('nan'),3],[0,0,400,4]):
            with self.assertRaises(ValueError):validate_box(box,32,32)

    def test_tracking_uses_new_frames_without_relocalizing(self):
        session=self.session();first=session.process(self.rgb,0)
        for frame in range(1,4):session.process(self.rgb+frame,frame)
        self.assertEqual(len(self.locator.calls),1)
        self.assertEqual(first.metadata['generation'],1)
        self.assertEqual(self.factory.instances[0].accepted,3)
        self.assertTrue(first.mask.any())

    def test_dropout_is_timed_and_never_filled_with_oracle(self):
        s=self.session(dict(intervention=dict(kind='dropout',start=2,duration=2)))
        frames=[s.process(self.rgb,i) for i in range(5)]
        self.assertEqual([bool(r.mask.any()) for r in frames],[True,True,False,False,True])
        self.assertTrue(frames[2].metadata['tracked_pixels']>0)
        self.assertFalse(frames[2].metadata['natural_lost'])
        self.assertEqual(frames[2].metadata['state'],'lost')
        self.assertEqual(len(self.locator.calls),1)

    def test_low_quality_frame_is_rejected_from_tracker_memory(self):
        mask=np.zeros((32,32),bool);mask[6:14,6:14]=True
        factory=Factory([(mask,.1)])
        s=self.session(factory=factory);s.process(self.rgb,0);bad=s.process(self.rgb,1)
        self.assertFalse(bad.mask.any());self.assertTrue(bad.metadata['natural_lost'])
        self.assertEqual(factory.instances[0].rejected,1)
        self.assertEqual(factory.instances[0].accepted,0)

    def test_bad_sam2_seed_never_enters_permanent_tracker_memory(self):
        class BadSegmenter:
            def __init__(self,full,quality):self.full,self.quality=full,quality
            def segment(self,rgb,box):
                mask=np.ones(rgb.shape[:2],bool) if self.full else np.eye(rgb.shape[0],dtype=bool)
                return mask,self.quality
        for full,quality in [(False,.1),(True,.99)]:
            factory=Factory()
            s=TrackingSession(self.locator,BadSegmenter(full,quality),factory,'cube')
            result=s.process(self.rgb,0)
            self.assertEqual(factory.instances,[])
            self.assertFalse(result.mask.any())
            self.assertEqual(result.metadata['generation'],0)

    def test_invalid_vlm_box_is_observed_loss_and_never_a_seed(self):
        class InvalidLocator:
            def locate(self,rgb,text):
                return dict(found=True,bbox_xyxy=[0,5,999,9],coordinate_space='pixels')
        factory=Factory()
        s=TrackingSession(InvalidLocator(),Segmenter(),factory,'cube')
        result=s.process(self.rgb,0)
        self.assertEqual(result.metadata['state'],'lost')
        self.assertFalse(factory.instances)
        self.assertTrue(any(e['type']=='vlm_bbox_invalid' for e in result.metadata['events']))

    def test_reacquisition_is_bounded_and_resets_generation(self):
        empty=np.zeros((32,32),bool)
        factory=Factory([(empty,0.)]*10)
        s=self.session(dict(loss_mode='reacquire',lost_grace=1,reacquire_cooldown=1,max_reacquires=1),factory)
        s.process(self.rgb,0);s.process(self.rgb,1);result=s.process(self.rgb,2)
        self.assertEqual(result.metadata['generation'],2)
        self.assertTrue(factory.instances[0].closed)
        self.assertEqual(len(self.locator.calls),2)
        s.process(self.rgb,3,force_reacquire=True)
        self.assertEqual(len(self.locator.calls),2)

    def test_shift_has_no_wraparound(self):
        mask=np.zeros((8,8),bool);mask[:,7]=True
        self.assertFalse(shift_mask(mask,2,0).any())

    def test_freeze_uses_previous_prediction_then_unfreezes(self):
        s=self.session(dict(intervention=dict(kind='freeze',start=1,duration=2)))
        first=s.process(self.rgb,0)
        self.assertTrue(np.array_equal(first.mask,s.process(self.rgb,1).mask))
        self.assertEqual(s.process(self.rgb,2).metadata['intervention'],'freeze')
        self.assertEqual(s.process(self.rgb,3).metadata['intervention'],'none')

    def test_duplicate_frame_is_cached_and_rewind_rejected(self):
        s=self.session();first=s.process(self.rgb,0)
        self.assertIs(s.process(self.rgb,0),first)
        s.process(self.rgb,2)
        with self.assertRaises(ValueError):s.process(self.rgb,1)

    def test_privileged_fields_cannot_cross_service_boundary(self):
        app=Application(self.locator,Segmenter(),self.factory)
        for forbidden in ('segmentation','target_idx','gt_mask','target_bbox','actor_id'):
            with self.assertRaises(ValueError):
                app.request('/v1/start',dict(image_png=encode_rgb(self.rgb),target_text='cube',**{forbidden:1}))
        with self.assertRaises(ValueError):
            app.request('/v1/start',dict(image_png=encode_rgb(self.rgb),target_text='cube',options={'gt_mask':[]}))

    def test_rgb_occlusion_changes_only_perception_image(self):
        s=self.session(dict(intervention=dict(kind='occlusion',start=1,duration=1)))
        before=self.rgb.copy();s.process(self.rgb,0)
        changed=s._image_for_perception(self.rgb,1)
        self.assertTrue(np.array_equal(self.rgb,before));self.assertFalse(np.array_equal(changed,before))


class HTTPContract(unittest.TestCase):
    def test_actual_client_session_and_mask_png_roundtrip(self):
        app=Application(Locator(),Segmenter(),Factory())
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.app=app
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        client=PerceptionMaskClient(f'http://127.0.0.1:{server.server_port}')
        try:
            rgb=np.full((32,32,3),140,np.uint8)
            self.assertTrue(client.health()['rgb_only'])
            result=client.start(rgb,'red cube',{'intervention':{'kind':'dropout','start':1,'duration':1}})
            self.assertEqual(result.mask.shape,(32,32))
            self.assertTrue(result.mask.any())
            self.assertFalse(client.step(rgb,1).mask.any())
            self.assertTrue(client.step(rgb,2).mask.any())
            client.close();self.assertFalse(app.sessions)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)


if __name__=='__main__':unittest.main()
