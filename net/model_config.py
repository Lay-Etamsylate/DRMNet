"""Reproducible architecture ablations, shared by training and evaluation."""

from copy import deepcopy


VARIANTS = ('baseline', 'mamba', 'freqfusion', 'fusion')


def build_model_config(base, variant='baseline'):
    if variant not in VARIANTS:
        raise ValueError('Unknown model variant: %s' % variant)
    cfg = deepcopy(base)
    cfg['model_variant'] = variant
    cfg['multidirection_stages'] = [3, 4] if variant in ('mamba', 'fusion') else []
    cfg['freqfusion_stages'] = [2] if variant in ('freqfusion', 'fusion') else []
    cfg.setdefault('fusion_checkpoint', True)
    cfg.setdefault('freqfusion_compressed_channels', 16)
    cfg.setdefault('freqfusion_kernel_size', 3)
    return cfg


def checkpoint_model_config(base, checkpoint, variant=None):
    """Restore saved architecture; untagged historical checkpoints are baseline.

    Explicitly reject a different variant rather than silently loading a partial
    network. Checkpoints lacking metadata can only be identified as baseline.
    """
    saved = checkpoint.get('model_config')
    saved_variant = saved.get('model_variant', 'baseline') if saved else 'baseline'
    if saved_variant not in VARIANTS:
        raise ValueError('Unknown checkpoint variant: %s' % saved_variant)
    if variant is not None and variant != saved_variant:
        raise ValueError('Checkpoint variant %r does not match requested %r' % (saved_variant, variant))
    cfg = build_model_config(base, saved_variant)
    if saved:
        cfg.update(deepcopy(saved))
    return cfg
