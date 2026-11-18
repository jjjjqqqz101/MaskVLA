"""Verify attention indexing and unchanged ACT predictions, without simulation.

Run with the mask-act environment. By default this checks both existing best
checkpoints on one validation observation using four CPU threads.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from act.attention_maps import AttentionRecorder, visual_token_indices


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def verify_synthetic():
    indices = visual_token_indices(2, 3, 2)
    expected = np.array([[[2, 3, 4], [8, 9, 10]], [[5, 6, 7], [11, 12, 13]]])
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(visual_token_indices(2, 3, 1), [[[2, 3, 4], [5, 6, 7]]])

    decoder = SimpleNamespace(return_intermediate=True, layers=[None] * 4)
    model = SimpleNamespace(training=False, transformer=SimpleNamespace(
        encoder=SimpleNamespace(layers=[None] * 2), decoder=decoder))
    recorder = AttentionRecorder(model)
    assert recorder.decoder_layer_index == 0
    model.transformer.decoder.return_intermediate = False
    assert AttentionRecorder(model).decoder_layer_index == 3
    model.transformer.decoder.return_intermediate = True

    # Two directed routes make rollout multiplication order observable:
    # layer 0: masked(1,2) -> RGB(0,1); layer 1: RGB(1,0) -> masked(1,2).
    first = np.eye(14, dtype=np.float32)
    second = first.copy()
    first[13] = 0
    first[13, 3] = 1
    second[8] = 0
    second[8, 13] = 1
    recorder.encoder_attention = [first[None], second[None]]
    recorder.decoder_attention = np.zeros((1, 3, 14), dtype=np.float32)
    recorder.decoder_attention[0, 0, 8] = 1
    recorder.decoder_attention[0, 1, 0] = 1
    recorder.decoder_attention[0, 2, 1] = 1
    recorder.feature_shape = (2, 3)
    recorder._feature_shapes = [(2, 3), (2, 3)]
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[1, 2] = 1
    maps = recorder.maps(2, mask, query_index=0, encoder_layer=0)

    expected_action = np.zeros(14, dtype=np.float32)
    expected_action[[8, 13, 3]] = [0.5, 0.25, 0.25]
    np.testing.assert_allclose(maps['action_token_mass'], expected_action, atol=1e-7)
    np.testing.assert_allclose(maps['action_original'], [[0, 0.25, 0], [0.5, 0, 0]])
    np.testing.assert_allclose(maps['action_masked'], [[0, 0, 0], [0, 0, 0.25]])
    np.testing.assert_allclose(maps['decoder_original'], [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(maps['mask_to_original'], [[0, 1, 0], [0, 0, 0]])
    assert maps['mask_relation_valid'] and maps['mask_to_original_mass'] == 1
    np.testing.assert_allclose(maps['encoder_rollout'].sum(axis=-1), 1)
    total_mass = sum(maps['action_mass_' + name] for name in ('latent', 'state', 'original', 'masked'))
    np.testing.assert_allclose(total_mass, 1)

    later = recorder.maps(2, mask, query_index=0, encoder_layer=1)
    np.testing.assert_array_equal(later['mask_to_original'], np.zeros((2, 3)))
    averaged = recorder.maps(2, mask, query_index=-1, encoder_layer=0)
    expected_average = expected_action.copy()
    expected_average[[0, 1]] = 1
    np.testing.assert_allclose(averaged['action_token_mass'], expected_average / 3, atol=1e-7)
    empty = recorder.maps(2, np.zeros_like(mask), query_index=0, encoder_layer=0)
    assert not empty['target_visible'] and not empty['mask_relation_valid']
    np.testing.assert_array_equal(empty['mask_to_original'], np.zeros((2, 3)))
    return dict(passed=True, width_concatenation=True, attention_direction=True,
                rollout_order=True, decoder_selection=True, empty_mask=True,
                query_selection=True, mass_conservation=True)


def load_observation(dataset, trajectory, frame):
    import h5py
    from torchvision.transforms import Resize
    from train_dual_rgb import STATE_FIELDS

    with h5py.File(dataset, 'r') as source:
        row = source[trajectory]
        rgb = np.stack([row['obs/sensor_data/base_camera/rgb'][frame],
                        row['derived/base_camera/masked_rgb'][frame]])
        state = np.concatenate([row[f'obs/{group}/{key}'][frame] for group, key in STATE_FIELDS])
        mask = np.asarray(row['derived/base_camera/target_mask'][frame]).squeeze()
    assert rgb.dtype == np.uint8 and state.shape == (28,) and mask.ndim == 2
    images = torch.from_numpy(rgb).permute(0, 3, 1, 2)
    images = Resize((224, 224), antialias=True)(images).unsqueeze(0)
    return images, torch.as_tensor(state, dtype=torch.float32)[None], mask


def hook_counts(model):
    return {name: len(module._forward_hooks) for name, module in model.named_modules()}


def verify_checkpoint(path, images, raw_state, mask, device):
    from evaluate_rgb_ablation import load_policy

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    agent, _, stats = load_policy(checkpoint, device)
    num_streams = len(checkpoint['stream_order'])
    observations = dict(rgb=images[:, :num_streams].to(device),
                        state=(raw_state.to(device) - stats['state_mean']) / stats['state_std'])

    def infer():
        # Agent.get_action mutates the image entry while normalizing it.
        return agent.get_action({name: value.clone() for name, value in observations.items()})

    original_hooks = hook_counts(agent.model)
    with torch.no_grad():
        baseline = infer()
        decoder_weights = {}
        handles = []

        def save_decoder(index):
            def capture(module, inputs, output):
                decoder_weights[index] = output[1].detach().float().cpu().numpy().copy()
            return capture

        try:
            layers = agent.model.transformer.decoder.layers
            for index in (0, len(layers) - 1):
                handles.append(layers[index].multihead_attn.register_forward_hook(save_decoder(index)))
            with AttentionRecorder(agent.model) as recorder:
                actual = infer()
                maps = recorder.maps(num_streams, mask, query_index=0, encoder_layer=0)
                assert recorder.decoder_layer_index == 0
                np.testing.assert_array_equal(recorder.decoder_attention, decoder_weights[0])
                if len(layers) > 1:
                    assert not np.allclose(recorder.decoder_attention, decoder_weights[len(layers) - 1])
        finally:
            for handle in handles:
                handle.remove()

        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        assert hook_counts(agent.model) == original_hooks
        try:
            with AttentionRecorder(agent.model):
                raise RuntimeError('intentional-hook-cleanup-check')
        except RuntimeError as error:
            assert str(error) == 'intentional-hook-cleanup-check'
        assert hook_counts(agent.model) == original_hooks
        torch.testing.assert_close(infer(), baseline, rtol=0, atol=0)

    for name, value in maps.items():
        if isinstance(value, (np.ndarray, float)):
            assert np.isfinite(value).all(), name
    np.testing.assert_allclose(maps['action_token_mass'].sum(), 1, atol=2e-5)
    total_mass = sum(maps['action_mass_' + name] for name in ('latent', 'state', 'original', 'masked'))
    np.testing.assert_allclose(total_mass, 1, atol=2e-5)
    assert maps['action_original'].shape == (7, 7)
    if num_streams == 2:
        assert maps['mask_relation_valid'] and maps['mask_to_original'].shape == (7, 7)
    else:
        assert 'mask_to_original' not in maps and not maps['mask_relation_valid']
    return dict(passed=True, checkpoint=str(path), streams=num_streams,
                max_prediction_difference=float((actual - baseline).abs().max()),
                decoder_layer=recorder.decoder_layer_index, hooks_removed=True,
                hooks_removed_on_exception=True, finite_maps=True, total_attention_mass=float(total_mass))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-only', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dataset', type=Path, default=REPO / 'demos/TargetPickCube-v1/motionplanning/target_pick_500_mask/dataset/validation.h5')
    parser.add_argument('--trajectory', default='traj_0')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--dual-checkpoint', type=Path, default=HERE / 'runs/TargetPickCube-v1__train_dual_rgb__1__001/checkpoints/best_eval_success_once.pt')
    parser.add_argument('--single-checkpoint', type=Path, default=HERE / 'runs/TargetPickCube-v1__train_single_rgb__1__1790243190/checkpoints/best_eval_success_once.pt')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    report = {'synthetic': verify_synthetic()}
    if not args.synthetic_only:
        images, state, mask = load_observation(args.dataset, args.trajectory, args.frame)
        report['checkpoints'] = [verify_checkpoint(path, images, state, mask, torch.device(args.device))
                                 for path in (args.dual_checkpoint, args.single_checkpoint)]
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
