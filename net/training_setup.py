"""Explicit warm starts versus exact training resumes."""

import math

from net.model_config import build_model_config, checkpoint_model_config

TRAINING_FIELDS = ('epochs', 'epoch_rcnn', 'optimizer', 'init_lr', 'momentum',
                   'weight_decay', 'max_grad_norm')


def configure_training_experiment(cfg, auxiliary_losses=None, scale_augmentation=None):
    """Explicit, checkpoint-persisted controls; absent flags preserve old behavior."""
    from copy import deepcopy
    result = deepcopy(cfg)
    for name, value in [('auxiliary_losses', auxiliary_losses), ('scale_augmentation', scale_augmentation)]:
        if value is not None and value not in ('on', 'off'):
            raise ValueError(name + ' must be on or off')
    if auxiliary_losses is not None:
        result['disable_auxiliary_losses'] = auxiliary_losses == 'off'
    if scale_augmentation is not None:
        result.setdefault('augtype', {})['scale'] = scale_augmentation == 'on'
    return result


def resolve_training_settings(defaults, resume=None, **overrides):
    settings = {key: defaults.get(key, 0.0 if key == 'max_grad_norm' else None)
                for key in TRAINING_FIELDS}
    if resume is not None:
        settings.update({key: value for key, value in resume.get('training_settings', {}).items()
                         if key in TRAINING_FIELDS})
    settings.update({key: value for key, value in overrides.items()
                     if key in TRAINING_FIELDS and value is not None})
    for key in ('epochs', 'epoch_rcnn'):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('init_lr', 'momentum', 'weight_decay', 'max_grad_norm'):
        value = settings[key]
        if value is None or not math.isfinite(value) or value < 0 or (key == 'init_lr' and value == 0):
            raise ValueError(f'{key} must be finite and nonnegative (init_lr must be positive)')
    return settings


def resolve_model_for_training(base, checkpoint=None, variant=None, head_activation=None,
                               resuming=False, reset_heads=False):
    if reset_heads and (checkpoint is None or resuming):
        raise ValueError('--reset-heads requires --init-from and cannot be used with --ckpt')
    cfg = (checkpoint_model_config(base, checkpoint, variant) if checkpoint is not None
           else build_model_config(base, variant or 'baseline'))
    saved_activation = cfg.get('head_activation', 'relu')
    if head_activation is not None:
        if resuming and head_activation != saved_activation:
            raise ValueError('Cannot change head activation during --ckpt resume; use --init-from --reset-heads')
        cfg['head_activation'] = head_activation
    cfg.setdefault('head_activation', 'relu')
    if cfg['head_activation'] not in ('relu', 'leaky_relu'):
        raise ValueError('head_activation must be relu or leaky_relu')
    return cfg


def reset_detection_heads(model, cfg):
    """Reset ONLY RPN and RCNN heads, keeping every backbone/ROI parameter."""
    from torch import nn
    slope = cfg.get('head_negative_slope', .1) if cfg.get('head_activation') == 'leaky_relu' else 0.
    for head in (model.rpn, model.rcnn_head):
        for module in head.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, a=slope, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    # Small nonzero output weights permit gradients into hidden layers from
    # the first update without reviving the old, very large RCNN regression.
    for module in (model.rpn.logits, model.rcnn_head.logit):
        nn.init.normal_(module.weight, std=.01)
        nn.init.zeros_(module.bias)
    for module in (model.rpn.deltas, model.rcnn_head.delta):
        nn.init.normal_(module.weight, std=.001)
        nn.init.zeros_(module.bias)


def load_training_weights(model, checkpoint, cfg, reset_heads=False):
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if reset_heads:
        reset_detection_heads(model, cfg)


def install_health_hooks(model, destination):
    """One-forward diagnostics; callers remove these handles immediately."""
    watched = [('rpn_hidden', model.rpn.conv[1])]
    if model.use_rcnn:
        watched.extend([('rcnn_fc1', model.rcnn_head.activation1),
                        ('rcnn_fc2', model.rcnn_head.activation2)])
    handles = []
    for name, module in watched:
        def hook(module, inputs, output, key=name):
            value = output.detach()
            destination[key + '_zero_fraction'] = float((value == 0).float().mean())
        handles.append(module.register_forward_hook(hook))
    return handles
