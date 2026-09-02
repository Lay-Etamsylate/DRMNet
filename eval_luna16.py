#!/usr/bin/env python3
"""Sliding-window DRMNet inference and official LUNA16 FROC evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

from config import config
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation
from net.sanet import SANet
from utils.pybox import py_nms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/data/home/zpc/jc/data/LUNA16")
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("/data/home/zpc/jc/data/LUNA16/drmnet_preprocessed"),
    )
    parser.add_argument("--filelist", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/luna16_eval"))
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--tile-stride", type=int, default=96)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--global-nms-threshold", type=float, default=0.1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip-froc", action="store_true")
    parser.add_argument("--max-scans", type=int, default=None)
    return parser.parse_args()


def json_ready(value):
    """Convert NumPy values returned by the legacy evaluator for JSON output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    return value


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_tile(
    image: np.ndarray,
    start_zyx: tuple[int, int, int],
    tile_size: int,
    pad_value: int,
) -> np.ndarray:
    z, y, x = start_zyx
    tile = image[
        z : z + tile_size, y : y + tile_size, x : x + tile_size
    ]
    padding = [(0, tile_size - tile.shape[axis]) for axis in range(3)]
    if any(after for _, after in padding):
        tile = np.pad(tile, padding, mode="constant", constant_values=pad_value)
    return tile


def model_outputs(net: SANet, tile: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    tensor = torch.from_numpy(tile[np.newaxis, np.newaxis].astype(np.float32))
    tensor = ((tensor - 128.0) / 128.0).to(device, non_blocking=True)
    with torch.inference_mode():
        net(tensor, None, None)

    outputs = {
        "rpn": net.rpn_proposals.detach().cpu().numpy(),
        "rcnn": net.detections.detach().cpu().numpy(),
        "ensemble": net.ensemble_proposals.detach().cpu().numpy(),
    }
    # Remove batch id and, for RCNN, the trailing class id.
    outputs["rpn"] = outputs["rpn"][:, 1:8]
    outputs["ensemble"] = outputs["ensemble"][:, 1:8]
    outputs["rcnn"] = outputs["rcnn"][:, 1:8]
    return outputs


def infer_scan(
    net: SANet,
    clean_path: Path,
    device: torch.device,
    tile_size: int,
    stride: int,
    score_threshold: float,
    nms_threshold: float,
) -> dict[str, np.ndarray]:
    clean = np.load(clean_path, mmap_mode="r")
    if clean.ndim != 4 or clean.shape[0] != 1:
        raise ValueError(f"Expected [1,D,H,W] at {clean_path}, got {clean.shape}")
    image = clean[0]
    starts = [tile_starts(size, tile_size, stride) for size in image.shape]
    collected = {"rpn": [], "rcnn": [], "ensemble": []}
    total = len(starts[0]) * len(starts[1]) * len(starts[2])
    index = 0
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                index += 1
                tile = extract_tile(image, (z, y, x), tile_size, config["pad_value"])
                outputs = model_outputs(net, tile, device)
                offset = np.asarray([z, y, x], dtype=np.float32)
                for name, detections in outputs.items():
                    if len(detections):
                        detections = detections.copy()
                        detections[:, 1:4] += offset
                        detections = detections[detections[:, 0] >= score_threshold]
                        if len(detections):
                            collected[name].append(detections)
                print(
                    f"\r  tile {index}/{total}",
                    end="",
                    flush=True,
                )
    print()

    merged = {}
    for name, parts in collected.items():
        if not parts:
            merged[name] = np.empty((0, 7), dtype=np.float32)
            continue
        detections = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        merged[name], _ = py_nms(detections, nms_threshold)
    return merged


def crop_detection_to_world_rows(
    uid: str, detections: np.ndarray, meta_path: Path
) -> list[dict]:
    if len(detections) == 0:
        return []
    with np.load(meta_path) as meta:
        crop_start_zyx = meta["crop_start_zyx"].astype(np.float64)
        origin_xyz = meta["resampled_origin_xyz"].astype(np.float64)
        spacing_xyz = meta["resampled_spacing_xyz"].astype(np.float64)
        direction = meta["resampled_direction"].astype(np.float64).reshape(3, 3)

    rows = []
    spacing_zyx = spacing_xyz[::-1]
    for detection in detections:
        score = float(detection[0])
        crop_center_zyx = detection[1:4].astype(np.float64)
        full_index_xyz = (crop_center_zyx + crop_start_zyx)[::-1]
        world_xyz = origin_xyz + direction @ (full_index_xyz * spacing_xyz)
        diameter_mm = float(np.mean(detection[4:7] * spacing_zyx))
        rows.append(
            {
                "seriesuid": uid,
                "coordX": world_xyz[0],
                "coordY": world_xyz[1],
                "coordZ": world_xyz[2],
                "diameter_mm": diameter_mm,
                "probability": score,
            }
        )
    return rows


def find_excluded_annotations(data_root: Path) -> Path:
    candidates = [
        data_root / "evaluationScript" / "annotations_excluded.csv",
        Path(__file__).resolve().parent
        / "evaluationScript"
        / "annotations_excluded.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find annotations_excluded.csv")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DRMNet inference")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    preprocessed_dir = args.preprocessed_dir.expanduser().resolve()
    filelist = args.filelist or preprocessed_dir / "filelists" / "test.npy"
    uids = [str(uid) for uid in np.load(filelist, allow_pickle=False)]
    if args.max_scans is not None:
        uids = uids[: args.max_scans]

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    net = SANet(config).to(device)
    net.load_state_dict(state_dict, strict=True)
    net.set_mode("eval")
    net.use_rcnn = True

    output_dir = args.output_dir.expanduser().resolve()
    detection_dir = output_dir / "detections"
    froc_dir = output_dir / "FROC"
    detection_dir.mkdir(parents=True, exist_ok=True)
    froc_dir.mkdir(parents=True, exist_ok=True)

    rows = {"rpn": [], "rcnn": [], "ensemble": []}
    for scan_index, uid in enumerate(uids, 1):
        print(f"[{scan_index}/{len(uids)}] {uid}")
        detections = infer_scan(
            net,
            preprocessed_dir / f"{uid}_clean.npy",
            device,
            args.tile_size,
            args.tile_stride,
            args.score_threshold,
            args.global_nms_threshold,
        )
        for name, stage_detections in detections.items():
            np.save(detection_dir / f"{uid}_{name}.npy", stage_detections)
            rows[name].extend(
                crop_detection_to_world_rows(
                    uid, stage_detections, preprocessed_dir / f"{uid}_meta.npz"
                )
            )

    columns = [
        "seriesuid",
        "coordX",
        "coordY",
        "coordZ",
        "diameter_mm",
        "probability",
    ]
    submissions = {}
    for name, stage_rows in rows.items():
        path = froc_dir / f"submission_{name}.csv"
        pd.DataFrame(stage_rows, columns=columns).to_csv(path, index=False)
        submissions[name] = path

    run_info = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "filelist": str(filelist),
        "num_scans": len(uids),
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
        "score_threshold": args.score_threshold,
        "global_nms_threshold": args.global_nms_threshold,
    }
    (output_dir / "run.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )

    if args.skip_froc or args.max_scans is not None:
        print(f"Submissions written to {froc_dir}; FROC was skipped.")
        return

    annotations = args.data_root / "annotations.csv"
    excluded = find_excluded_annotations(args.data_root)
    seriesuids = preprocessed_dir / "filelists" / "test_seriesuids.csv"
    for name, submission in submissions.items():
        stage_output = froc_dir / name
        stage_output.mkdir(exist_ok=True)
        sens_boot, froc_boot, sens, froc = noduleCADEvaluation(
            str(annotations),
            str(excluded),
            str(seriesuids),
            str(submission),
            str(stage_output),
        )
        metrics = {
            "sensitivity_bootstrap": sens_boot,
            "froc_bootstrap": froc_boot,
            "sensitivity": sens,
            "froc": froc,
        }
        (stage_output / "metrics.json").write_text(
            json.dumps(json_ready(metrics), indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: FROC={froc}, bootstrap FROC={froc_boot}")


if __name__ == "__main__":
    main()
