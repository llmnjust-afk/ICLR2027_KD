"""Strict compatibility checks for upstream DiT weights and local identity adapters."""
import argparse
import json
import re

import torch


GAMMA = re.compile(r'blocks\.[0-9]+\.(?:attn\.)?gamma[12]')
MODE_EMBEDDING = 'm_embedder.embedding_table.weight'


def unwrap(state):
    if isinstance(state, dict) and isinstance(state.get('ema'), dict):
        return state['ema']
    if isinstance(state, dict) and isinstance(state.get('model'), dict):
        return state['model']
    if not isinstance(state, dict):
        raise ValueError('DiT checkpoint must contain a state dictionary.')
    return state


def validate(model, state, mode_embedding_unused=True, check_identity=True):
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    invalid = [key for key in missing if not GAMMA.fullmatch(key)
               and not (key == MODE_EMBEDDING and mode_embedding_unused)]
    shapes = [key for key in set(expected) & set(state)
              if not isinstance(state[key], torch.Tensor) or state[key].shape != expected[key].shape]
    if unexpected or invalid or shapes:
        raise ValueError(f'Checkpoint incompatible with DiT: missing_required={invalid}, '
                         f'unexpected={unexpected}, shape_mismatch={sorted(shapes)}')
    if check_identity:
        for key in missing:
            if GAMMA.fullmatch(key):
                value = expected[key]
                if value.is_meta or not bool(torch.all(value == 1)):
                    raise ValueError(f'Missing gamma must already be initialized to identity: {key}')
    return {'missing_initialized_identity_or_unused': missing, 'unexpected': unexpected,
            'validated_checkpoint_tensors': len(state), 'mode_embedding_unused': mode_embedding_unused}


def load_compatible(model, state, mode_embedding_unused=True):
    state = unwrap(state)
    report = validate(model, state, mode_embedding_unused)
    model.load_state_dict(state, strict=False)
    return report


def inspect_checkpoint(path):
    from models import DiT_models
    # Metadata-only on the login node; real CPU tensor values are checked during loading.
    with torch.device('meta'):
        model = DiT_models['DiT-XL/2'](input_size=32, num_classes=1000)
    state = unwrap(torch.load(path, map_location='meta'))
    return validate(model, state, check_identity=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2))


if __name__ == '__main__':
    main()
