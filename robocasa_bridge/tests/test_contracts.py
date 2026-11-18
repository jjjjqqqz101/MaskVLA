import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,h5py
from robocasa_bridge.scene import validate_order
from robocasa_bridge.paths import ROOT,episode_dir
from robocasa_bridge.engine import Engine
from robocasa_bridge.training import validate_raw,batch,openpi_sample

class Contracts(unittest.TestCase):
    def test_order_requires_exact_two_targets(self):
        self.assertEqual(validate_order(["blue_cube","red_cube"]),["blue_cube","red_cube"])
        for invalid in (["red_cube"],["red_cube","red_cube"],["other","blue_cube"]):
            with self.assertRaises(ValueError):validate_order(invalid)

    def test_episode_id_has_no_path_escape(self):
        for invalid in ("../secret","rc_../bad","/mnt/else","rc_x/y","rc_"):
            with self.assertRaises(ValueError):episode_dir(invalid)

    def test_meter_commands_map_to_native_fixed_base(self):
        def convert(d):
            return np.r_[d["right"],d["right_gripper"],d["base"],d["torso"],d["base_mode"]]
        e=Engine(lambda *args:None,SimpleNamespace(is_set=lambda:False))
        e.env=SimpleNamespace(robots=[SimpleNamespace(create_action_vector=convert)])
        native,arm=e.native_action([.01,-.02,0],[0,0,.05],1)
        np.testing.assert_allclose(arm,[.2,-.4,0,0,0,.1,1])
        np.testing.assert_array_equal(native[7:11],0)
        self.assertEqual(native[-1],-1)
        with self.assertRaises(ValueError):e.native_action([.051,0,0],[0,0,0],1)

    def test_ordered_score_does_not_accept_wrong_first_target(self):
        e=Engine(lambda *args:None,SimpleNamespace(is_set=lambda:False))
        e.env=object();e.order=["red_cube","blue_cube"]
        placed={"red_cube":False,"blue_cube":True}
        e._placed=lambda name:placed[name]
        for _ in range(3):e.update_score()
        self.assertFalse(e.order_valid)
        self.assertEqual(e.completed,["blue_cube"])
        placed["red_cube"]=True
        for _ in range(3):e.update_score()
        self.assertFalse(e.scoring()["simulator_success"])

    def test_reverse_order_is_a_supported_task(self):
        e=Engine(lambda *args:None,SimpleNamespace(is_set=lambda:False))
        e.env=object();e.order=["blue_cube","red_cube"]
        placed={"red_cube":False,"blue_cube":True}
        e._placed=lambda name:placed[name]
        for _ in range(3):e.update_score()
        placed["red_cube"]=True
        for _ in range(3):e.update_score()
        self.assertTrue(e.order_valid)
        self.assertTrue(e.scoring()["simulator_success"])

    def test_current_scene_restore_normalizes_recorded_phase_only(self):
        e=Engine(lambda *args:None,SimpleNamespace(is_set=lambda:False))
        e.env=object()
        e.initial={"stage":"manual_recording","frame_id":0}
        e.initial_rgb_hashes={"base_rgb":"base","wrist_rgb":"wrist","overview_rgb":"overview"}
        e.scene_info={}
        restored=[]
        def restore(snap):
            restored.append(snap)
            e.stage=snap["stage"]
            e.frame_id=snap["frame_id"]
        e.restore_snapshot=restore
        e.rgb_hashes=lambda:dict(e.initial_rgb_hashes)
        published=[]
        e.publish=lambda:published.append(e.stage)
        e.scene({"restore":True})
        self.assertEqual(e.stage,"scene_preview")
        self.assertEqual(published,["scene_preview"])
        self.assertEqual(restored,[e.initial])
        self.assertEqual(e.initial["stage"],"manual_recording")
        self.assertTrue(e.scene_info["initial_verified"])
        self.assertEqual(e.frame_id,0)
        self.assertIsNone(e.record)

    def test_scene_restore_phase_fix_keeps_strict_rgb_rejection(self):
        e=Engine(lambda *args:None,SimpleNamespace(is_set=lambda:False))
        e.env=object();e.initial={"stage":"manual_recording"}
        e.initial_rgb_hashes={"base_rgb":"expected"};e.scene_info={}
        e.restore_snapshot=lambda snap:setattr(e,"stage",snap["stage"])
        e.rgb_hashes=lambda:{"base_rgb":"wrong"}
        published=[];e.publish=lambda:published.append(e.stage)
        with self.assertRaisesRegex(RuntimeError,"RGB verification"):
            e.scene({"restore":True})
        self.assertEqual(published,[])
        self.assertNotIn("initial_verified",e.scene_info)

    def fixture(self):
        root=ROOT/"work/robocasa_bridge/tests";root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.tmp.cleanup)
        p=Path(self.tmp.name)/"raw.h5";t=3
        m=dict(order=["red_cube","blue_cube"],policy_state_dim=23,
               action_contract={"native_parts":{"right":[0,6],"right_gripper":[6,7],"base":[7,10],"torso":[10,11]}})
        with h5py.File(p,"w") as f:
            f.attrs["metadata_json"]=json.dumps(m);o=f.create_group("observations");a=f.create_group("actions")
            base=np.full((4,224,224,3),[30,60,90],np.uint8);wrist=np.full_like(base,150)
            mask=np.zeros((4,224,224),np.uint8);mask[:,20:30,40:50]=1
            o["base_rgb"]=base;o["wrist_rgb"]=wrist;o["gt_target_mask_offline"]=mask
            o["robot_state"]=np.arange(4*23).reshape(4,23).astype(np.float32)
            o["subtask_index"]=[0,0,1,2]
            o.create_dataset("target_id",data=["red_cube","red_cube","blue_cube",""],dtype=h5py.string_dtype())
            actions=np.arange(3*7).reshape(3,7).astype(np.float32)*.01
            a["arm_gripper"]=actions
            a["native"]=np.c_[actions,np.zeros((3,4)),-np.ones(3)]
        return p

    def test_action_alignment_and_chunk_padding(self):
        p=self.fixture();data,_,_=batch(p,[0,2],"act_triple",horizon=4,normalize=False)
        self.assertEqual(data["rgb"].shape,(2,3,3,224,224))
        self.assertEqual(data["state"].shape,(2,23))
        np.testing.assert_array_equal(data["is_pad"],[[False,False,True,True],[False,True,True,True]])
        with h5py.File(p,"r") as f:
            np.testing.assert_array_equal(data["actions"][0,:2],f["actions/arm_gripper"][:2])
            np.testing.assert_array_equal(data["actions"][1,0],f["actions/arm_gripper"][2])

    def test_mask_is_single_frame_cutout_and_stream_order(self):
        data,_,_=batch(self.fixture(),[0],"act_triple",normalize=False)
        self.assertTrue(np.all(data["rgb"][0,0,:,0,0]==[30,60,90]))
        self.assertTrue(np.all(data["rgb"][0,1,:,0,0]==150))
        self.assertTrue(np.all(data["rgb"][0,2,:,0,0]==0))
        self.assertTrue(np.all(data["rgb"][0,2,:,25,45]==[30,60,90]))

    def test_openpi_uses_new_23d_robot_not_old_8d(self):
        row,_,_=openpi_sample(self.fixture())
        self.assertEqual(row["observation/state"].shape,(23,))
        self.assertEqual(row["actions"].shape,(10,7))
        self.assertIn("red cube",row["prompt"])

    def test_broken_T_plus_one_rejected(self):
        p=self.fixture()
        with h5py.File(p,"a") as f:
            del f["observations/robot_state"];f["observations/robot_state"]=np.zeros((3,23))
        with self.assertRaisesRegex(ValueError,"alignment"):validate_raw(p)

    def test_wrong_subtask_target_rejected(self):
        p=self.fixture()
        with h5py.File(p,"a") as f:f["observations/target_id"][2]="red_cube"
        with self.assertRaisesRegex(ValueError,"identity"):validate_raw(p)

if __name__=="__main__":unittest.main()
