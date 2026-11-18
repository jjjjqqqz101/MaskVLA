"""Dependency-free validation of reproducible initial-scene references."""
import copy
import unittest

from task_agent.scene_reference import validate_scene_reference


def reference():
    return dict(source_scene_id='task_agent_reference_fixture',seed=17,
                rgb_sha256={'base_rgb':'a'*64,'wrist_rgb':'b'*64,'overview_rgb':'c'*64})


class SceneReferenceContracts(unittest.TestCase):
    def test_valid_reference_is_copied_and_hex_is_normalized(self):
        value=reference();value['rgb_sha256']['base_rgb']='A'*64
        result=validate_scene_reference(value,17)
        self.assertEqual(result['rgb_sha256']['base_rgb'],'a'*64)
        result['rgb_sha256']['wrist_rgb']='d'*64
        self.assertEqual(value['rgb_sha256']['wrist_rgb'],'b'*64)

    def test_unknown_and_missing_fields_or_nonobjects_are_rejected(self):
        for value in (None,False,[],dict(reference(),extra=True),
                      {k:v for k,v in reference().items() if k!='seed'}):
            with self.subTest(value=value),self.assertRaises(ValueError):validate_scene_reference(value)

    def test_id_is_a_single_existing_console_identifier_format(self):
        for identifier in ('','other','../task_agent_bad','task_agent_a/b','task_agent_'+('a'*101),None):
            value=reference();value['source_scene_id']=identifier
            with self.subTest(identifier=identifier),self.assertRaises(ValueError):validate_scene_reference(value)

    def test_seed_is_strict_and_expected_seed_must_match(self):
        for seed in (True,17.,'17',-1,2**31,None):
            value=reference();value['seed']=seed
            with self.subTest(seed=seed),self.assertRaises(ValueError):validate_scene_reference(value)
        for expected in (18,True,-1):
            with self.subTest(expected=expected),self.assertRaises(ValueError):validate_scene_reference(reference(),expected)

    def test_exact_camera_hash_keys_and_hex_lengths_are_required(self):
        for hashes in ({'base_rgb':'a'*64},{**reference()['rgb_sha256'],'extra':'d'*64},[],None):
            value=reference();value['rgb_sha256']=hashes
            with self.subTest(hashes=hashes),self.assertRaises(ValueError):validate_scene_reference(value)
        for digest in ('a'*63,'a'*65,'g'*64,' '*64,17,True):
            value=copy.deepcopy(reference());value['rgb_sha256']['base_rgb']=digest
            with self.subTest(digest=digest),self.assertRaises(ValueError):validate_scene_reference(value)


if __name__=='__main__':unittest.main()
