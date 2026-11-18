import json,tempfile,unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,h5py
from robocasa_bridge import catalog
from robocasa_bridge.paths import ROOT
from robocasa_bridge.scene import validate_order
from robocasa_bridge.engine import Engine
from robocasa_bridge.training import batch,openpi_sample
import test_contracts as old_contracts

class SelectionContracts(unittest.TestCase):
    def setUp(self):
        root=ROOT/"work/robocasa_bridge/catalog_tests";root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.tmp.cleanup)
        self.assets=Path(self.tmp.name)
        for name,size in (("apple_0",".04 .03 .04"),("apple_1",".05 .04 .05")):
            p=self.assets/"objects/objaverse/apple"/name;p.mkdir(parents=True)
            (p/"mesh.obj").write_text("mock mesh")
            (p/"model.xml").write_text('<mujoco><asset><mesh name="m" file="mesh.obj"/></asset><worldbody><body><body name="object"><geom name="reg_bbox" type="box" size="'+size+'" pos="0 0 0"/></body></body></worldbody></mujoco>')
        p=self.assets/"objects/lightwheel/stool/Stool001";p.mkdir(parents=True);(p/"model.xml").write_text("<mujoco/>")
        self.legacy_reference = "/legacy/asset_export/ceramic.png"
        self.patches=patch.multiple(catalog,ASSETS=self.assets,_CACHE=None,_CACHE_TIME=0.,
            LEGACY_CERAMIC_PATH_SHA256=hashlib.sha256(self.legacy_reference.encode()).hexdigest(),
            _REGISTRY_CACHE={"apple":{"types":"fruit","graspable":True,"objaverse":{"scale":.9}}})
        self.patches.start();self.addCleanup(self.patches.stop)

    def test_catalog_has_real_instances_and_disabled_decor(self):
        c=catalog.get_catalog(refresh=True)
        self.assertEqual(c["downloaded_instances"],3)
        apple=next(x for x in c["categories"] if x["name"]=="apple")
        self.assertEqual(apple["instance_count"],2)
        self.assertTrue(apple["selectable"])
        stool=next(x for x in c["categories"] if x["name"]=="stool")
        self.assertFalse(stool["selectable"])
        variants=catalog.get_catalog(category_id=apple["catalog_id"])["instances"]
        self.assertEqual(len(variants),2)
        self.assertNotIn("path",variants[0])
        self.assertEqual(catalog.get_catalog(query="苹果")["total"],1)

    def test_seed_and_explicit_instance_produce_stable_concrete_selection(self):
        req=[{"catalog_id":"objaverse:apple","quantity":2}]
        a=catalog.resolve_selection(req,42);b=catalog.resolve_selection(req,42)
        self.assertEqual(a,b)
        self.assertNotEqual(a[0]["id"],a[1]["id"])
        self.assertEqual(len(a[0]["model_sha256"]),64)
        exact=catalog.resolve_selection([{"catalog_id":"objaverse:apple","instance_id":a[0]["instance_id"]}],7)
        self.assertEqual(a[0]["model_path"],exact[0]["model_path"])
        self.assertEqual(catalog.validate_resolved(a),a)

    def test_invalid_size_count_path_and_missing_mesh_rejected(self):
        for req in ([],[{"catalog_id":"objaverse:apple","quantity":7}],[{"catalog_id":"objaverse:apple","quantity":True}],
                    [{"catalog_id":"objaverse:apple","instance_id":"../x"}]):
            with self.assertRaises(ValueError):catalog.resolve_selection(req,1)
        with self.assertRaises(ValueError):catalog.asset_path(ROOT/"repo")
        variants=catalog.catalog_index()["objaverse:apple"]["_instances"]
        v=variants[0];(v["path"].parent/"mesh.obj").unlink()
        request={"catalog_id":"objaverse:apple","instance_id":v["instance_id"]}
        with self.assertRaisesRegex(ValueError,"no suitable"):catalog.resolve_selection([request],1)
        s=v["path"].read_text().replace('.04 .03 .04','4 3 4');v["path"].write_text(s)
        info=catalog.instance_info(catalog.catalog_index()["objaverse:apple"],v)
        self.assertFalse(info["selectable"])

    def test_one_and_six_targets_and_target_override_are_not_task_success(self):
        targets=[f"item_{i:02d}_apple" for i in range(6)]
        self.assertEqual(validate_order(None,targets),targets)
        self.assertEqual(validate_order(None,targets[:1]),targets[:1])
        with self.assertRaises(ValueError):validate_order([targets[0],targets[0]],targets[:2])
        e=Engine(lambda *a:None,SimpleNamespace(is_set=lambda:False))
        e.env=object();e.order=targets;e.completed=[targets[0]]
        before=(list(e.completed),e.order_valid,e.frame_id)
        calls=[];e.publish=lambda:calls.append(e.target())
        e.select_target({"target_id":targets[5]})
        self.assertEqual(e.target(),targets[5])
        self.assertEqual((e.completed,e.order_valid,e.frame_id),before)
        e.select_target({"target_id":None})
        self.assertIsNone(e.active_subtask)
        self.assertEqual(e.target(),targets[1])
        self.assertEqual(calls,[targets[5],targets[1]])

    def test_generic_three_targets_keep_chunk_boundaries_and_prompt_labels(self):
        path=old_contracts.Contracts.fixture(self)
        ids=["item_01_apple","item_02_banana","item_03_mug"]
        with h5py.File(path,"a") as f:
            m=json.loads(f.attrs["metadata_json"]);m["order"]=ids
            m["selected_objects"]=[{"id":x,"name":n} for x,n in zip(ids,("apple","banana","mug"))]
            f.attrs["metadata_json"]=json.dumps(m)
            f["observations/subtask_index"][:]=[0,1,2,3]
            f["observations/target_id"][:]=ids+[""]
        data,_,_=batch(path,[0,1,2],horizon=3,normalize=False)
        np.testing.assert_array_equal(data["is_pad"],[[False,True,True]]*3)
        pi,_,_=openpi_sample(path,1)
        self.assertIn("the banana",pi["prompt"])
        self.assertNotIn("item 02",pi["prompt"])


    def test_exact_aigen_legacy_texture_is_relocated_and_recorded(self):
        shared=self.assets/catalog.SHARED_CERAMIC
        shared.parent.mkdir(parents=True);shared.write_bytes(b"official-test-texture")
        digest=hashlib.sha256(shared.read_bytes()).hexdigest()
        item=catalog.catalog_index()["objaverse:apple"]["_instances"][0]
        path=item["path"]
        xml=path.read_text().replace("<asset>","<asset><texture name='ceramic' type='2d' file='"+self.legacy_reference+"'/>")
        path.write_text(xml)
        original=path.read_bytes()
        with patch.object(catalog,"CERAMIC_SHA256",digest):
            self.assertEqual(catalog.reference_path(path,self.legacy_reference,"texture"),shared)
            selected=catalog.resolve_selection([{"catalog_id":"objaverse:apple","instance_id":item["instance_id"]}],1)[0]
            relocation=selected["asset_reference_relocations"][0]
            self.assertEqual(relocation["sha256"],digest)
            self.assertEqual(relocation["local_asset_path"],catalog.SHARED_CERAMIC)
            self.assertEqual(path.read_bytes(),original)

    def test_legacy_relocation_does_not_accept_other_external_paths_or_changed_texture(self):
        item=catalog.catalog_index()["objaverse:apple"]["_instances"][0]
        path=item["path"]
        for forbidden in ("/external/ceramic.png",self.legacy_reference.replace("asset_export","other"),
                          "../../../../../../outside.png"):
            with self.assertRaises(ValueError):catalog.reference_path(path,forbidden,"texture")
        with self.assertRaisesRegex(ValueError,"only for a texture"):
            catalog.reference_path(path,self.legacy_reference,"mesh")
        shared=self.assets/catalog.SHARED_CERAMIC
        shared.parent.mkdir(parents=True);shared.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError,"checksum"):
            catalog.reference_path(path,self.legacy_reference,"texture")

if __name__=="__main__":unittest.main()
