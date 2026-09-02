#!/usr/bin/env python3
"""Run one synthetic DRMNet training step to validate the remote environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# When this file is executed as ``python tools/smoke_test_model.py``, Python
# places ``tools/`` rather than the repository root on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import config
from net.sanet import SANet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    model = SANet(config).to(device)
    model.set_mode("train")
    model.use_rcnn = False
    inputs = torch.zeros((1, 1, 128, 128, 128), device=device)
    truth_boxes = np.asarray(
        [np.asarray([[64.0, 64.0, 64.0, 10.0, 10.0, 10.0]], dtype=np.float32)],
        dtype=object,
    )
    truth_labels = np.asarray(
        [np.asarray([1.0], dtype=np.float32)], dtype=object
    )

    model(inputs, truth_boxes, truth_labels)
    loss, rpn_stats, _ = model.loss()
    if args.backward:
        loss.backward()
    print("device:", torch.cuda.get_device_name(args.device))
    print("loss:", float(loss.detach().cpu()))
    print("rpn_stats:", rpn_stats)
    print("backward:", args.backward)
    print("DRMNet smoke test passed")


if __name__ == "__main__":
    main()
