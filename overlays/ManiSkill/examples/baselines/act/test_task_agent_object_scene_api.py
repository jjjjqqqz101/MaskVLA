"""Frozen inventory HTTP contracts; fake workers only, no model/simulator."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_agent.scene_objects import CLASSIC_OBJECT_IDS, catalog_public, normalize_scene_spec, resolve_scene_spec
from task_agent.web import Console, Handler


HASHES = dict(base_rgb='0'*64, wrist_rgb='1'*64, overview_rgb='a'*64)


class FakeProcess:
    def __init__(self): self.returncode = None
    def poll(self): return self.returncode


class ProcessFactory:
    def __init__(self): self.calls = []; self.processes = []; self.spec_at_spawn = []
    def __call__(self, command, **options):
        folder = Path(command[command.index('--output')+1])
        self.spec_at_spawn.append(json.loads((folder/'scene_spec.json').read_text()))
        process = FakeProcess()
        self.calls.append((command, options)); self.processes.append(process)
        return process


class FirstSampler:
    def sample(self, population, count): return list(population)[:count]


class QuietHandler(Handler):
    def log_message(self, *args): pass


class ObjectSceneAPIContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.factory = ProcessFactory()
        self.console = Console(Path(self.temp.name)/'runs', process_factory=self.factory)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        self.server.console = self.console
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:%d' % self.server.server_port

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.temp.cleanup()

    def get(self, route):
        with urlopen(self.url+route, timeout=3) as response: return json.load(response)

    def post(self, route, payload):
        request = Request(self.url+route, data=json.dumps(payload, allow_nan=True).encode(),
            headers={'Content-Type': 'application/json', 'X-CSRF-Token': self.console.token})
        try:
            with urlopen(request, timeout=3) as response: return response.status, json.load(response)
        except HTTPError as error: return error.code, json.load(error)

    def prepare(self, objects=None, seed=17):
        payload = dict(seed=seed)
        if objects is not None: payload['objects'] = objects
        code, response = self.post('/api/scene', payload)
        self.assertEqual(code, 202, response)
        return self.get('/api/status')['scene']

    def folder(self, scene): return self.console.directory(scene['id'])

    def manifest(self, scene, spec=None, include_spec=True, **extra):
        expected = json.loads((self.folder(scene)/'scene_spec.json').read_text()) if spec is None else spec
        value = dict(scene_id=scene['id'], seed=scene['seed'], reset_count=1, rgb_sha256=dict(HASHES))
        if include_spec:
            value.update(scene_spec=expected, selected_object_ids=[r['id'] for r in expected['objects']])
        value.update(extra)
        (self.folder(scene)/'scene.json').write_text(json.dumps(value))
        (self.folder(scene)/'status.json').write_text(json.dumps(dict(status='scene_ready', phase='scene_ready',
            scene_id=scene['id'], seed=scene['seed'])))
        return self.get('/api/status')['scene']

    def finish(self, scene):
        (self.folder(scene)/'summary.json').write_text(json.dumps(dict(status='stopped', success=False, seed=scene['seed'])))
        self.factory.processes[-1].returncode = 0

    def test_public_catalog_exposes40_entries_and_training_coverage(self):
        response = self.get('/api/object_catalog')
        self.assertTrue(response['ok'])
        original = catalog_public()
        self.assertEqual([{k:v for k,v in item.items() if k != 'act_triple_trained'}
                          for item in response['objects']], original)
        self.assertTrue(all(item['act_triple_trained'] == item['pi05_trained']
                            for item in response['objects']))
        self.assertEqual(sum(item['act_triple_trained'] for item in response['objects']), 25)
        self.assertEqual((response['min_objects'], response['max_objects']), (1, 8))
        self.assertEqual(response['classic_object_ids'], list(CLASSIC_OBJECT_IDS))

    def test_old_prepare_payload_keeps_classic_order_and_freezes_spec_before_worker_spawn(self):
        scene = self.prepare()
        expected = normalize_scene_spec()
        self.assertEqual(self.factory.spec_at_spawn, [expected])
        self.assertEqual(scene['scene_spec'], expected)
        self.assertEqual(scene['selected_object_ids'], list(CLASSIC_OBJECT_IDS))
        self.assertEqual(len(scene['object_names']), 8)
        self.assertEqual(json.loads((self.folder(scene)/'web_request.json').read_text())['scene_spec'], expected)

    def test_manual_ids_order_and_scales_are_saved_exactly_and_projected_in_status(self):
        ids = ['056_tennis_ball', '025_mug', '013_apple']
        choice = dict(mode='manual', object_ids=ids)
        scene = self.prepare(choice)
        expected = resolve_scene_spec(choice)
        self.assertEqual(scene['scene_spec'], expected)
        self.assertEqual(scene['selected_object_ids'], ids)
        self.assertEqual(self.factory.spec_at_spawn, [expected])
        request = json.loads((self.folder(scene)/'web_request.json').read_text())
        self.assertEqual(request['object_selection'], choice)
        self.assertEqual(self.get('/api/status')['run']['scene_spec'], expected)
        self.assertEqual(next(r for r in self.get('/api/runs')['runs'] if r['id'] == scene['id'])['selected_object_ids'], ids)
        self.assertEqual(scene['scene_spec']['objects'][1]['scale'], .7)

    def test_same_fixed_seed_random_refresh_changes_contents_not_only_positions(self):
        choice = dict(mode='random', count=8, pool='train')
        with patch('task_agent.scene_objects.random.SystemRandom', return_value=FirstSampler()):
            first = self.prepare(choice, seed=17)
            self.manifest(first)
            self.finish(first)
            second = self.prepare(choice, seed=17)
        self.assertEqual((first['seed'], second['seed']), (17, 17))
        self.assertNotEqual(set(first['selected_object_ids']), set(second['selected_object_ids']))
        train = {r['id'] for r in catalog_public() if r['pi05_trained']}
        self.assertTrue(set(second['selected_object_ids']) <= train)
        self.assertEqual(len(set(second['selected_object_ids'])), 8)

    def test_invalid_selection_cannot_start_a_worker_or_write_a_partial_scene(self):
        invalid = [dict(mode='manual', object_ids=[]), dict(mode='manual', object_ids=['missing']),
            dict(mode='manual', object_ids=['025_mug', '025_mug']),
            dict(mode='manual', object_ids=['025_mug'], scale=1),
            dict(mode='random', count=True, pool='all'), dict(mode='random', count=9, pool='all'),
            dict(mode='random', count=0, pool='train'), dict(mode='random', count=3, pool='heldout')]
        for choice in invalid:
            code, response = self.post('/api/scene', dict(seed=17, objects=choice))
            self.assertEqual(code, 400, response)
        self.assertEqual(self.factory.calls, [])
        self.assertEqual(list(self.console.runs_root.iterdir()), [])

    def test_custom_ready_requires_worker_report_of_the_same_ordered_inventory(self):
        choice = dict(mode='manual', object_ids=['013_apple', '025_mug'])
        scene = self.prepare(choice)
        self.assertFalse(self.manifest(scene, include_spec=False)['ready'])
        reordered = resolve_scene_spec(dict(mode='manual', object_ids=['025_mug', '013_apple']))
        self.assertFalse(self.manifest(scene, spec=reordered)['ready'])
        self.assertFalse(self.manifest(scene, selected_object_ids=['013_apple'])['ready'])
        code, response = self.post('/api/scene/task', dict(scene_id=scene['id'], instruction='收拾水果'))
        self.assertEqual(code, 409, response)
        self.assertFalse((self.folder(scene)/'task_request.json').exists())
        self.assertTrue(self.manifest(scene)['ready'])

    def test_task_submission_cannot_change_inventory_after_the_preview(self):
        scene = self.prepare(dict(mode='manual', object_ids=['013_apple']))
        self.manifest(scene)
        original = (self.folder(scene)/'scene_spec.json').read_bytes()
        code, response = self.post('/api/scene/task', dict(scene_id=scene['id'], instruction='收拾水果',
            objects=dict(mode='manual', object_ids=['025_mug'])))
        self.assertEqual(code, 400, response)
        self.assertEqual((self.folder(scene)/'scene_spec.json').read_bytes(), original)
        self.assertEqual(len(self.factory.calls), 1)

    def test_comparison_copies_custom_inventory_without_extending_reference_schema(self):
        choice = dict(mode='manual', object_ids=['056_tennis_ball', '025_mug', '013_apple'])
        source = self.prepare(choice)
        self.manifest(source)
        self.finish(source)
        original = {p.name: p.read_bytes() for p in self.folder(source).iterdir()}
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source['id'], objects=dict(mode='classic')))
        self.assertEqual(code, 400, response)
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source['id']))
        self.assertEqual(code, 202, response)
        clone = self.get('/api/status')['scene']
        self.assertEqual(clone['scene_spec'], source['scene_spec'])
        self.assertEqual(self.factory.spec_at_spawn[-1], source['scene_spec'])
        reference = json.loads((self.folder(clone)/'scene_reference.json').read_text())
        self.assertEqual(set(reference), {'source_scene_id', 'seed', 'rgb_sha256'})
        self.assertEqual(reference['source_scene_id'], source['id'])
        self.assertEqual({p.name: p.read_bytes() for p in self.folder(source).iterdir()}, original)
        self.assertFalse(self.manifest(clone, comparison_initial_layout_verified=True,
            comparison_source_scene_id=source['id'], selected_object_ids=list(CLASSIC_OBJECT_IDS))['ready'])
        self.assertTrue(self.manifest(clone, comparison_initial_layout_verified=True,
            comparison_source_scene_id=source['id'])['ready'])

    def test_tampered_or_symlinked_frozen_custom_spec_cannot_be_comparison_source(self):
        source = self.prepare(dict(mode='manual', object_ids=['013_apple']))
        self.manifest(source)
        self.finish(source)
        path = self.folder(source)/'scene_spec.json'
        original = path.read_bytes()
        path.write_text(json.dumps(resolve_scene_spec(dict(mode='manual', object_ids=['025_mug']))))
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source['id']))
        self.assertEqual(code, 400, response)
        external = Path(self.temp.name)/'outside_scene_spec.json'
        external.write_bytes(original)
        path.unlink(); path.symlink_to(external)
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source['id']))
        self.assertEqual(code, 400, response)
        self.assertEqual(len(self.factory.calls), 1)

    def test_legacy_classic_reference_without_a_spec_keeps_original_eight(self):
        source_id = 'task_agent_legacy_classic'
        folder = self.console.runs_root/source_id; folder.mkdir()
        (folder/'web_request.json').write_text(json.dumps(dict(mode='scene_first', scene_id=source_id, seed=17)))
        (folder/'scene.json').write_text(json.dumps(dict(scene_id=source_id, seed=17, reset_count=1, rgb_sha256=HASHES)))
        (folder/'summary.json').write_text(json.dumps(dict(status='stopped', success=False)))
        code, response = self.post('/api/scene', dict(mode='current', source_scene_id=source_id))
        self.assertEqual(code, 202, response)
        self.assertEqual(self.get('/api/status')['scene']['scene_spec'], normalize_scene_spec())
        self.assertEqual(self.factory.spec_at_spawn, [normalize_scene_spec()])


if __name__ == '__main__': unittest.main()
