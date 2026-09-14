"""Detection-oriented 3D adaptations of MedSegMamba and FreqFusion ideas.

These are independent implementations, not the upstream models/checkpoints.
See docs/fusion.md for sources, design differences and validation limits.
"""

from itertools import product

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def make_mamba(**kwargs):
    # Keep pure-PyTorch filtering and scan-layout tests usable without CUDA.
    from mamba_ssm import Mamba
    return Mamba(**kwargs)


def volume_to_sequence(x, order, reverse=False):
    """B,C,D,H,W -> B,L,C, with an explicit spatial traversal order."""
    sequence = x.permute(0, *(axis + 2 for axis in order), 1).reshape(
        x.shape[0], -1, x.shape[1]
    )
    return sequence.flip(1) if reverse else sequence


def sequence_to_volume(sequence, shape, order, reverse=False):
    """Undo traversal, including reverse, before merging spatial features."""
    if reverse:
        sequence = sequence.flip(1)
    x = sequence.reshape(sequence.shape[0], *(shape[i] for i in order), -1)
    return x.permute(0, 4, *(order.index(i) + 1 for i in range(3))).contiguous()


class MultiDirectionMamba3d(nn.Module):
    """Three axis-priority traversals, each forward/backward, plus local detail.

    Traversals share a Mamba within a block. Different encoder stages receive
    independent blocks. This is not MedSegMamba's fixed-size snake/Hilbert scan.
    ``mamba_factory`` is an injection point for layout tests, never a fallback.
    """

    ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))

    def __init__(self, channels, d_state=16, d_conv=4, expand=2,
                 use_checkpoint=True, mamba_factory=None):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.norm = nn.LayerNorm(channels)
        factory = mamba_factory or make_mamba
        self.ssm = factory(d_model=channels, d_state=d_state,
                           d_conv=d_conv, expand=expand)
        self.direction_logits = nn.Parameter(torch.zeros(6))
        self.local = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 1),
        )
        self.out_norm = nn.LayerNorm(channels)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 0.01))

    def _scan_direction(self, x, order, reverse):
        tokens = volume_to_sequence(x, order, reverse)
        # Selective-state dynamics run in FP32 even under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self.ssm(self.norm(tokens.float()).contiguous())
        return sequence_to_volume(y, x.shape[2:], order, reverse)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError("MultiDirectionMamba3d expects B,C,D,H,W with matching channels")
        weights = self.direction_logits.softmax(0)
        merged = None
        for i, (order, reverse) in enumerate(product(self.ORDERS, (False, True))):
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                y = checkpoint(self._scan_direction, x, order, reverse,
                               use_reentrant=False)
            else:
                y = self._scan_direction(x, order, reverse)
            contribution = y * weights[i]
            merged = contribution if merged is None else merged + contribution
        merged = self.out_norm(merged.movedim(1, -1)).movedim(-1, 1)
        residual = merged.to(x.dtype) + self.local(x)
        return x + self.scale.to(x.dtype) * residual


def adaptive_filter3d(x, weights, kernel_size):
    """Per-voxel, channel-shared filter without a B*C*K^3*D*H*W unfold.

    Replicate padding preserves constants, including at volume boundaries.
    The caller supplies normalized nonnegative weights, shaped B,K^3,D,H,W.
    Shifted views share storage; avoid materializing every channel/neighbour.
    """
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    expected = (x.shape[0], kernel_size ** 3, *x.shape[2:])
    if x.ndim != 5 or tuple(weights.shape) != expected:
        raise ValueError("filter weights must have shape B,K^3,D,H,W")
    radius = kernel_size // 2
    padded = F.pad(x, (radius,) * 6, mode='replicate')
    d, h, w = x.shape[2:]
    result = None
    for i, (z, y, xx) in enumerate(product(range(kernel_size), repeat=3)):
        shifted = padded[:, :, z:z + d, y:y + h, xx:xx + w]
        contribution = shifted * weights[:, i:i + 1].to(x.dtype)
        result = contribution if result is None else result + contribution
    return result


class FreqFusion3d(nn.Module):
    """Simplified adaptive low/high-pass refinement inspired by FreqFusion.

    Refines decoder/skip features before concatenation. It does not implement
    CARAFE, offsets, or the upstream iterative mask-generation procedure.
    Channels and the skip's spatial resolution are preserved.
    """

    def __init__(self, hr_channels=64, lr_channels=64, compressed_channels=16,
                 kernel_size=3, use_checkpoint=True):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = kernel_size
        self.use_checkpoint = use_checkpoint
        self.hr_compress = nn.Conv3d(hr_channels, compressed_channels, 1)
        self.lr_compress = nn.Conv3d(lr_channels, compressed_channels, 1)
        self.lowpass_logits = nn.Conv3d(compressed_channels, kernel_size ** 3, 3, padding=1)
        self.highpass_logits = nn.Conv3d(compressed_channels, kernel_size ** 3, 3, padding=1)
        # Bounded per-voxel gates start with a modest filtering contribution.
        self.gates = nn.Conv3d(compressed_channels, 2, 1)
        window = torch.hamming_window(kernel_size, periodic=False)
        window = (window[:, None, None] * window[None, :, None]
                  * window[None, None, :]).reshape(1, -1, 1, 1, 1)
        self.register_buffer('window', window)
        for generator in (self.lowpass_logits, self.highpass_logits):
            nn.init.normal_(generator.weight, std=0.001)
            nn.init.zeros_(generator.bias)
        nn.init.zeros_(self.gates.weight)
        nn.init.constant_(self.gates.bias, -2.0)

    def _weights(self, logits):
        weights = logits.float().softmax(dim=1) * self.window.float()
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _forward(self, hr_feat, lr_feat):
        if lr_feat.shape[2:] != hr_feat.shape[2:]:
            lr_feat = F.interpolate(lr_feat, size=hr_feat.shape[2:],
                                    mode='trilinear', align_corners=False)
        context = F.silu(self.hr_compress(hr_feat) + self.lr_compress(lr_feat))
        lowpass = self._weights(self.lowpass_logits(context))
        highpass = self._weights(self.highpass_logits(context))
        low_gate, high_gate = self.gates(context).sigmoid().chunk(2, dim=1)
        smooth_lr = adaptive_filter3d(lr_feat, lowpass, self.kernel_size)
        smooth_hr = adaptive_filter3d(hr_feat, highpass, self.kernel_size)
        lr_refined = lr_feat + low_gate * (smooth_lr - lr_feat)
        hr_refined = hr_feat + high_gate * (hr_feat - smooth_hr)
        return hr_refined, lr_refined

    def forward(self, hr_feat, lr_feat):
        if hr_feat.ndim != 5 or lr_feat.ndim != 5 or hr_feat.shape[0] != lr_feat.shape[0]:
            raise ValueError("FreqFusion3d expects two B,C,D,H,W tensors with the same batch size")
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, hr_feat, lr_feat, use_reentrant=False)
        return self._forward(hr_feat, lr_feat)
