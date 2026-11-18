"""CPU-only deployment paths and launcher contracts; no service/model startup."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import deployment_paths as paths

PATH_VARIABLES = (
    'MASK_POLICY_ROOT', 'MASK_ACT_CHECKPOINT', 'MASK_ACT_TRIPLE_CHECKPOINT',
    'MASK_PI05_CHECKPOINT', 'MASK_DATA_ROOT', 'MASK_TASK_RUNS_ROOT',
    'PERCEPTION_RUNS_ROOT', 'SAM2_CHECKPOINT', 'CUTIE_CHECKPOINT',
    'SAM2_REPO', 'CUTIE_REPO', 'TORCH_HOME', 'OPENPI_DATA_HOME', 'MS_ASSET_DIR', 'QWEN_CONFIG',
)


class DeploymentPaths(unittest.TestCase):
    def clean_environment(self):
        return {key: value for key, value in os.environ.items() if key not in PATH_VARIABLES}

    def temporary_directory(self):
        base = paths.DEFAULT_ROOT / 'runtime/tmp'
        base.mkdir(parents=True, exist_ok=True)
        return TemporaryDirectory(dir=base)

    def test_default_root_follows_source_checkout(self):
        parent = paths.MANISKILL_ROOT.parent
        expected = parent.parent if parent.name == 'repo' else parent
        with patch.dict(os.environ, self.clean_environment(), clear=True):
            self.assertEqual(paths.policy_root(), expected)

    def test_all_resources_follow_deployment_root(self):
        with self.temporary_directory() as folder, patch.dict(os.environ, self.clean_environment(), clear=True):
            os.environ['MASK_POLICY_ROOT'] = folder
            expected = {
                'act_checkpoint': 'ckpt/maniskill/act_dual_rgb_8x100/100000.pt',
                'act_triple_checkpoint': 'ckpt/maniskill/act_triple_rgb_25x50/160000.pt',
                'pi05_checkpoint': 'ckpt/maniskill/pi05_25x50/49999',
                'data_root': 'data/maniskill/pi05_train_25x50',
                'task_runs_root': 'outputs/task_agent',
                'perception_runs_root': 'outputs/perception_tracking',
                'sam2_checkpoint': 'ckpt/perception/sam2.1_hiera_small.pt',
                'cutie_checkpoint': 'ckpt/perception/cutie-base-mega.pth',
                'sam2_repo': 'repo/third_party/sam2',
                'cutie_repo': 'repo/third_party/Cutie',
                'torch_cache': 'ckpt/maniskill/support/torch',
                'openpi_cache': 'ckpt/maniskill/support/openpi',
                'assets_root': 'assets/maniskill',
                'qwen_config': 'config/private/qwen.json',
            }
            for name, relative in expected.items():
                with self.subTest(resource=name):
                    self.assertEqual(getattr(paths, name)(), Path(folder).resolve() / relative)

    def test_individual_paths_override_root_at_call_time(self):
        functions = (
            ('MASK_ACT_CHECKPOINT', paths.act_checkpoint),
            ('MASK_ACT_TRIPLE_CHECKPOINT', paths.act_triple_checkpoint),
            ('MASK_PI05_CHECKPOINT', paths.pi05_checkpoint),
            ('MASK_DATA_ROOT', paths.data_root),
            ('MASK_TASK_RUNS_ROOT', paths.task_runs_root),
            ('PERCEPTION_RUNS_ROOT', paths.perception_runs_root),
            ('SAM2_CHECKPOINT', paths.sam2_checkpoint),
            ('CUTIE_CHECKPOINT', paths.cutie_checkpoint),
            ('SAM2_REPO', paths.sam2_repo),
            ('CUTIE_REPO', paths.cutie_repo),
            ('TORCH_HOME', paths.torch_cache),
            ('OPENPI_DATA_HOME', paths.openpi_cache),
            ('MS_ASSET_DIR', paths.assets_root),
            ('QWEN_CONFIG', paths.qwen_config),
        )
        with self.temporary_directory() as folder, patch.dict(os.environ, self.clean_environment(), clear=True):
            os.environ['MASK_POLICY_ROOT'] = folder
            for index, (variable, function) in enumerate(functions):
                with self.subTest(variable=variable):
                    expected = Path(folder).resolve() / f'override-{index}'
                    os.environ[variable] = str(expected)
                    self.assertEqual(function(), expected)

    def test_empty_override_uses_deployment_default(self):
        with self.temporary_directory() as folder, patch.dict(os.environ, self.clean_environment(), clear=True):
            os.environ.update(MASK_POLICY_ROOT=folder, SAM2_REPO='')
            self.assertEqual(paths.sam2_repo(), Path(folder).resolve() / 'repo/third_party/sam2')

    def test_project_paths_reject_external_roots_overrides_and_symlink_escape(self):
        with patch.dict(os.environ, self.clean_environment(), clear=True):
            with self.assertRaises(ValueError): paths.project_path('/tmp/outside-mask-policy')
            with patch.dict(os.environ, {'MASK_POLICY_ROOT': '/tmp/outside-mask-policy'}):
                with self.assertRaises(ValueError): paths.policy_root()
            with patch.dict(os.environ, {'MASK_ACT_CHECKPOINT': '/tmp/outside-mask-policy.pt'}):
                with self.assertRaises(ValueError): paths.act_checkpoint()
            with self.temporary_directory() as folder:
                link = Path(folder) / 'escape'
                link.symlink_to('/tmp', target_is_directory=True)
                with self.assertRaises(ValueError): paths.project_path(link / 'outside-mask-policy')

    def test_qwen_config_environment_is_read_without_external_fallback(self):
        from agentic_act.qwen import QwenClient
        with self.temporary_directory() as folder, patch.dict(os.environ, self.clean_environment(), clear=True):
            config = Path(folder) / 'qwen.json'
            config.write_text(json.dumps({'api_key': 'test-only-credential',
                'base_url': 'https://example.invalid/compatible-mode/v1', 'model': 'test-model'}))
            os.environ.update(QWEN_CONFIG=str(config), QWEN_API_KEY='', DASHSCOPE_API_KEY='',
                              QWEN_BASE_URL='', QWEN_MODEL='')
            client = QwenClient()
            self.assertEqual(client.model, 'test-model')
            self.assertEqual(client.key, 'test-only-credential')
            with patch.object(Path, 'read_text', side_effect=AssertionError('Must reject before reading')):
                with self.assertRaises(ValueError): QwenClient(config='/tmp/outside-qwen.json')

    def test_launchers_inherit_runtime_and_preserve_arguments(self):
        for script, python_variable, module in (
            ('run_task_agent_web.sh', 'MASK_ACT_PYTHON', 'task_agent.web'),
            ('run_perception_mask.sh', 'PERCEPTION_PYTHON', 'perception_mask.server'),
        ):
            with self.subTest(script=script), self.temporary_directory() as folder:
                executable = Path(folder) / 'fake-python'
                executable.write_text(f'#!{sys.executable}\n' +
                    'import json, os, sys\n' +
                    'print(json.dumps({"argv":sys.argv[1:], "root":os.environ["MASK_POLICY_ROOT"],'
                    ' "torch":os.environ["TORCH_HOME"], "assets":os.environ["MS_ASSET_DIR"],'
                    '  "qwen":os.environ["QWEN_CONFIG"]}))\n')
                executable.chmod(0o755)
                env = self.clean_environment()
                for variable in ('MASK_VENV_ROOT', 'MASK_ACT_PYTHON', 'PERCEPTION_PYTHON', 'OPENPI_PYTHON'):
                    env.pop(variable, None)
                env.update(MASK_POLICY_ROOT=str(paths.DEFAULT_ROOT), **{python_variable: str(executable)})
                process = subprocess.run(['bash', str(paths.ACT_ROOT / script), '--port', '19999'],
                                         env=env, text=True, capture_output=True, check=True)
                result = json.loads(process.stdout)
                self.assertEqual(result['argv'][:3], ['-B', '-m', module])
                self.assertEqual(result['argv'][-2:], ['--port', '19999'])
                self.assertEqual(result['root'], str(paths.DEFAULT_ROOT))
                self.assertEqual(result['torch'], str(paths.DEFAULT_ROOT / 'ckpt/maniskill/support/torch'))
                self.assertEqual(result['assets'], str(paths.DEFAULT_ROOT / 'assets/maniskill'))
                self.assertEqual(result['qwen'], str(paths.DEFAULT_ROOT / 'config/private/qwen.json'))
                if script == 'run_task_agent_web.sh':
                    self.assertIn(str(paths.DEFAULT_ROOT / 'outputs/task_agent'), result['argv'])


if __name__ == '__main__':
    unittest.main()
