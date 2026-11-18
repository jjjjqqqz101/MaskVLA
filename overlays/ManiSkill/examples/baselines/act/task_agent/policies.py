"""Stable public policy identities, without checkpoint paths or model imports.

The legacy ``act`` ID always identifies the eight-object dual-RGB model. A new
checkpoint family must receive a new ID so saved comparisons stay meaningful.
"""
import copy
import re


POLICY_IDS = ('act', 'act_triple', 'pi05')
POLICY_LABELS = {
    'act': '双路 ACT · 100000 · 8目标',
    'act_triple': '三路 ACT · 160000 · 25目标',
    'pi05': 'π0.5 · 49999 · 25×50',
}
_MODELS = {
    'act': dict(model_id='act_dual_rgb_8x100_100000', checkpoint_step=100000,
                rgb_streams=['base_rgb', 'base_masked_rgb'],
                rgb_stream_labels=['BaseRGB', 'BaseMaskedRGB'], state_dim=28,
                training_object_count=8, training_episodes_per_object=100),
    'act_triple': dict(model_id='act_triple_rgb_25x50_160000', checkpoint_step=160000,
                rgb_streams=['base_rgb', 'wrist_rgb', 'base_masked_rgb'],
                rgb_stream_labels=['BaseRGB', 'WristRGB', 'BaseMaskedRGB'], state_dim=28,
                training_object_count=25, training_episodes_per_object=50),
    'pi05': dict(model_id='pi05_25x50_49999', checkpoint_step=49999,
                rgb_streams=['base_rgb', 'wrist_rgb', 'base_masked_rgb'],
                rgb_stream_labels=['BaseRGB', 'WristRGB', 'BaseMaskedRGB'], state_dim=8,
                training_object_count=25, training_episodes_per_object=50),
}


def policy_metadata(policy):
    """Return an independent public model description for a known policy ID."""
    if not isinstance(policy, str) or policy not in POLICY_IDS:
        raise ValueError('Unknown skill policy')
    return dict(policy=policy, policy_label=POLICY_LABELS[policy], **copy.deepcopy(_MODELS[policy]))


def policy_record(policy, recorded_model=None):
    """Use saved public model provenance, falling back to the stable ID registry.

    Only known public fields are copied from disk. In particular, backend
    checkpoint paths and credentials never become model configuration output.
    A mismatched policy in old metadata cannot relabel a legacy ACT run.
    """
    if not isinstance(policy, str) or policy not in POLICY_IDS:
        return dict(policy=policy, policy_label=str(policy or '未选择'), model=None)
    model = policy_metadata(policy)
    if isinstance(recorded_model, dict) and recorded_model.get('policy') == policy:
        identifier = recorded_model.get('model_id')
        if isinstance(identifier, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', identifier):
            # Persisting complete public snapshots keeps older model versions
            # readable if the display registry is extended in the future.
            for key in model:
                value = recorded_model.get(key)
                if key == 'model_id':
                    model[key] = identifier
                elif key in ('checkpoint_step', 'state_dim', 'training_object_count', 'training_episodes_per_object'):
                    if type(value) is int and value >= 0:
                        model[key] = value
                elif key in ('rgb_streams', 'rgb_stream_labels'):
                    if (isinstance(value, list) and 1 <= len(value) <= 8
                            and all(isinstance(item, str) and re.fullmatch(r'[A-Za-z0-9_]{1,40}', item)
                                    for item in value)):
                        model[key] = list(value)
                elif key == 'policy_label' and isinstance(value, str) and 1 <= len(value) <= 120:
                    model[key] = value
    return dict(policy=policy, policy_label=model['policy_label'], model=model)


def policy_options():
    return [policy_record(policy) for policy in POLICY_IDS]
