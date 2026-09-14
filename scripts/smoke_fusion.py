"""Run real CUDA Mamba + feature pyramid + RPN head forward/backward.

This checks the model interface and measures this synthetic workload only;
it does not measure scan-level detection recall or proposal/RCNN/NMS cost.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from net.feature_net import FeatureNet
from net.heads import RpnHead
from net.model_config import VARIANTS, build_model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, default='fusion')
    parser.add_argument('--shape', nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-checkpoint', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('An NVIDIA CUDA device is required; this script has no simulated backend')
    if any(n < 16 or n % 16 for n in args.shape):
        parser.error('--shape dimensions must be positive multiples of 16')
    if args.steps < 1 or args.batch_size < 1:
        parser.error('--steps and --batch-size must be positive')
    if args.batch_size == 1 and all(n == 16 for n in args.shape):
        parser.error('training BatchNorm needs >1 value at the deepest stage; increase shape or batch size')
    cfg = build_model_config({'anchors': [[s, s, s] for s in (5, 10, 20, 30, 50)]}, args.variant)
    cfg['fusion_checkpoint'] = not args.no_checkpoint
    torch.manual_seed(35202)
    feature_net = FeatureNet(cfg).cuda().train()
    head = RpnHead(cfg).cuda().train()
    x = torch.randn(args.batch_size, 1, *args.shape, device='cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    optimizer = torch.optim.SGD(list(feature_net.parameters()) + list(head.parameters()), lr=1e-4)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
            features, _ = feature_net(x)
            logits, deltas = head(features[-1])
            expected = (args.batch_size, *(n // 4 for n in args.shape), 5)
            assert tuple(logits.shape) == (*expected, 1)
            assert tuple(deltas.shape) == (*expected, 6)
            loss = logits.float().square().mean() + deltas.float().square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite synthetic loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in feature_net.named_parameters():
            if 'multidirection.' in name or 'freqfusion.' in name:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError('Missing/non-finite gradient: ' + name)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    step()  # warm up kernels and allocate gradients before timing
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize()
    print(json.dumps({
        'variant': args.variant,
        'device': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'shape': args.shape,
        'batch_size': args.batch_size,
        'amp': args.amp,
        'checkpoint': cfg['fusion_checkpoint'],
        'parameters_feature_net_and_rpn': sum(p.numel() for p in feature_net.parameters()) + sum(p.numel() for p in head.parameters()),
        'seconds_per_synthetic_training_step': (time.perf_counter() - start) / args.steps,
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'last_synthetic_loss': loss,
        'scope': 'FeatureNet + RPN only; excludes proposals, RCNN, NMS and data loading',
    }, indent=2))


if __name__ == '__main__':
    main()
