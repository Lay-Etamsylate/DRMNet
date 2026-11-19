"""CPU numerical/layout tests; CUDA tests below use the real Mamba backend.

The injected causal mixer is ONLY a test double: CPU integration tests do not
claim to exercise the Mamba CUDA kernels or the detector's proposal/NMS code.
"""

import copy
import importlib.util
from itertools import product

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from net.feature_net import FeatureNet
from net.fusion_modules import (
    MultiDirectionMamba3d, FreqFusion3d, adaptive_filter3d,
    volume_to_sequence, sequence_to_volume,
)
from net.heads import RpnHead, RcnnHead
from net.model_config import VARIANTS, build_model_config, checkpoint_model_config


torch.set_num_threads(2)


class CausalTestMixer(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return self.projection(x).cumsum(1) / x.shape[1]


BASE = {'anchors': [[5, 5, 5], [10, 10, 10]],
        'num_class': 2, 'rcnn_crop_size': (7, 7, 7)}


@pytest.mark.parametrize('order,reverse', list(product(MultiDirectionMamba3d.ORDERS, (False, True))))
def test_scan_coordinates_and_gradients(order, reverse):
    # Non-cubic, non-contiguous volumes expose bad inverse permutations.
    x = torch.randn(1, 4, 2, 3, 5).transpose(2, 4).requires_grad_()
    block = MultiDirectionMamba3d(4, mamba_factory=CausalTestMixer)
    sequence = volume_to_sequence(x, order, reverse)
    restored = sequence_to_volume(sequence, x.shape[2:], order, reverse)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)
    restored.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))

    # Independent coordinate-by-coordinate oracle for a causal traversal.
    coords = list(product(*(range(x.shape[axis + 2]) for axis in order)))
    if reverse:
        coords.reverse()
    expected = torch.zeros_like(x)
    total = torch.zeros(1, 4)
    for coord in coords:
        canonical = tuple(coord[order.index(axis)] for axis in range(3))
        index = (slice(None), slice(None), *canonical)
        total = total + block.ssm.projection(block.norm(x[index]))
        expected[index] = total / len(coords)
    torch.testing.assert_close(block._scan_direction(x, order, reverse), expected)


def test_adaptive_filter_matches_explicit_neighbourhood_and_gradcheck():
    x = torch.randn(1, 2, 2, 3, 4, dtype=torch.double, requires_grad=True)
    weights = torch.randn(1, 27, 2, 3, 4, dtype=torch.double).softmax(1).requires_grad_()
    neighbours = F.pad(x, (1,) * 6, mode='replicate').unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1)
    expected = (neighbours.reshape(1, 2, 2, 3, 4, 27)
                * weights.movedim(1, -1).unsqueeze(1)).sum(-1)
    torch.testing.assert_close(adaptive_filter3d(x, weights, 3), expected)
    assert torch.autograd.gradcheck(lambda a, b: adaptive_filter3d(a, b, 3),
                                    (x, weights), fast_mode=True)


def test_frequency_fusion_preserves_constants_and_handles_odd_sizes():
    module = FreqFusion3d(hr_channels=3, lr_channels=5, compressed_channels=4)
    hr = torch.full((2, 3, 5, 7, 9), 2.0)
    lr = torch.full((2, 5, 3, 4, 5), -3.0)
    high, low = module(hr, lr)
    torch.testing.assert_close(high, hr)
    torch.testing.assert_close(low, torch.full((2, 5, 5, 7, 9), -3.0))


def test_2x_reassembly_matches_independent_child_weights_and_gradients():
    x = torch.randn(2, 3, 2, 1, 3, dtype=torch.double, requires_grad=True)
    weights = torch.randn(2, 27, 4, 2, 6, dtype=torch.double).softmax(1).requires_grad_()
    padded = F.pad(x, (1,) * 6, mode='replicate')
    expected = torch.zeros(2, 3, 4, 2, 6, dtype=x.dtype)
    for index, (z, y, xx) in enumerate(product(range(3), repeat=3)):
        shifted = padded[:, :, z:z+2, y:y+1, xx:xx+3]
        expected += F.interpolate(shifted, scale_factor=2, mode='nearest') * weights[:, index:index+1]
    actual = adaptive_filter3d(x, weights, 3, scale_factor=2)
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(lambda a, b: adaptive_filter3d(a, b, 3, scale_factor=2),
                                    (x, weights), fast_mode=True)


def test_cross_scale_variant_fuses_native_pyramid_levels_and_restores_checkpoint():
    cfg = build_model_config(BASE, 'fusion_cross_scale')
    torch.manual_seed(12)
    reference = FeatureNet(build_model_config(BASE, 'fusion'), mamba_factory=CausalTestMixer)
    reference_state = reference.state_dict()
    torch.manual_seed(12)
    model = FeatureNet(cfg, mamba_factory=CausalTestMixer).eval()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference_state[key], rtol=0, atol=0)
    levels = []
    def record_levels(module, inputs):
        levels.append([tuple(x.shape[2:]) for x in inputs])
    model.freqfusion['2'].register_forward_pre_hook(record_levels)
    x = torch.randn(1, 1, 16, 32, 48)
    features, _ = model(x)
    assert levels == [[(4, 8, 12), (2, 4, 6)]]
    restored = FeatureNet(checkpoint_model_config(BASE, {'model_config': cfg}),
                          mamba_factory=CausalTestMixer).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x)[0][-1], features[-1])


def test_frequency_branches_smooth_and_enhance_an_impulse():
    module = FreqFusion3d(1, 1, compressed_channels=2, use_checkpoint=False)
    for generator in (module.lowpass_logits, module.highpass_logits):
        nn.init.zeros_(generator.weight)
        nn.init.zeros_(generator.bias)
    x = torch.zeros(1, 1, 5, 5, 5)
    x[:, :, 2, 2, 2] = 1
    high, low = module(x, x)
    assert high[0, 0, 2, 2, 2] > 1
    assert 0 < low[0, 0, 2, 2, 2] < 1
    assert low[0, 0, 2, 2, 1] > 0


@pytest.mark.parametrize('kind', ['mamba', 'freqfusion'])
def test_checkpoint_matches_forward_and_parameter_gradients(kind):
    if kind == 'mamba':
        a = MultiDirectionMamba3d(4, use_checkpoint=False, mamba_factory=CausalTestMixer)
    else:
        a = FreqFusion3d(4, 4, compressed_channels=3, use_checkpoint=False)
    b = copy.deepcopy(a)
    b.use_checkpoint = True
    inputs = [torch.randn(2, 4, 3, 4, 5, requires_grad=True)
              for _ in range(1 if kind == 'mamba' else 2)]
    inputs_b = [t.detach().clone().requires_grad_() for t in inputs]
    out_a, out_b = a(*inputs), b(*inputs_b)
    if kind == 'mamba':
        out_a, out_b = (out_a,), (out_b,)
    for first, second in zip(out_a, out_b):
        torch.testing.assert_close(first, second)
    sum(x.square().mean() for x in out_a).backward()
    sum(x.square().mean() for x in out_b).backward()
    for first, second in zip(inputs, inputs_b):
        torch.testing.assert_close(first.grad, second.grad)
    for (name, first), (_, second) in zip(a.named_parameters(), b.named_parameters()):
        assert first.grad is not None, name
        assert torch.isfinite(first.grad).all(), name
        torch.testing.assert_close(first.grad, second.grad)


@pytest.mark.parametrize('variant', VARIANTS)
def test_feature_and_detector_head_contracts_with_test_mixer(variant):
    cfg = build_model_config(BASE, variant)
    feature_net = FeatureNet(cfg, mamba_factory=CausalTestMixer).eval()
    x = torch.randn(1, 1, 16, 32, 48)
    features, roi_skip = feature_net(x)
    assert [tuple(t.shape) for t in features] == [
        (1, 64, 8, 16, 24), (1, 64, 8, 16, 24), (1, 128, 4, 8, 12)]
    assert roi_skip is features[1]
    logits, deltas = RpnHead(cfg)(features[-1])
    assert logits.shape == (1, 4, 8, 12, 2, 1)
    assert deltas.shape == (1, 4, 8, 12, 2, 6)
    crops = F.adaptive_max_pool3d(roi_skip, cfg['rcnn_crop_size'])
    rcnn_logits, rcnn_deltas = RcnnHead(cfg, in_channels=64)(crops)
    assert rcnn_logits.shape == (1, 2)
    assert rcnn_deltas.shape == (1, 12)
    (logits.square().mean() + deltas.square().mean()
     + rcnn_logits.square().mean() + rcnn_deltas.square().mean()).backward()
    for name, parameter in feature_net.named_parameters():
        if 'multidirection.' in name or 'freqfusion.' in name:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_stage_blocks_are_independent_and_baseline_has_no_new_parameters():
    baseline = FeatureNet(build_model_config(BASE), mamba_factory=CausalTestMixer)
    assert not any('multidirection.' in k or 'freqfusion.' in k for k in baseline.state_dict())
    fusion = FeatureNet(build_model_config(BASE, 'fusion'), mamba_factory=CausalTestMixer)
    assert fusion.resnet50.multidirection['3'].ssm is not fusion.resnet50.multidirection['4'].ssm
    with pytest.raises(RuntimeError):
        fusion.load_state_dict(baseline.state_dict(), strict=True)


def test_configuration_checkpoint_restores_variant_and_custom_settings():
    original = copy.deepcopy(BASE)
    cfg = build_model_config(BASE, 'fusion')
    cfg['freqfusion_kernel_size'] = 5
    restored = checkpoint_model_config(BASE, {'model_config': cfg})
    assert restored == cfg
    assert BASE == original
    restored['anchors'][0][0] = 99
    assert cfg['anchors'][0][0] == 5
    assert checkpoint_model_config(BASE, {})['model_variant'] == 'baseline'
    with pytest.raises(ValueError, match='does not match'):
        checkpoint_model_config(BASE, {'model_config': cfg}, 'baseline')
    with pytest.raises(ValueError):
        checkpoint_model_config(BASE, {}, 'fusion')


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec('mamba_ssm') is None,
                    reason='requires NVIDIA CUDA and mamba_ssm; CPU tests use a test mixer')
def test_real_cuda_mamba_forward_backward_and_amp():
    for enabled in (False, True):
        model = MultiDirectionMamba3d(16).cuda().train()
        x = torch.randn(2, 16, 3, 4, 5, device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=enabled):
            y = model(x)
            loss = y.square().mean()
        loss.backward()
        assert torch.isfinite(y).all()
        assert torch.isfinite(x.grad).all()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
