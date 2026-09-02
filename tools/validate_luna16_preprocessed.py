#!/usr/bin/env python3
"""Validate DRMNet-preprocessed LUNA16 data before starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "pid",
    "zmin",
    "zmax",
    "ymin",
    "ymax",
    "xmin",
    "xmax",
    "coordX",
    "coordY",
    "coordZ",
    "diameter_mm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("/data/home/zpc/jc/data/LUNA16/drmnet_preprocessed"),
    )
    parser.add_argument("--expected-scans", type=int, default=888)
    parser.add_argument("--expected-annotations", type=int, default=1186)
    parser.add_argument(
        "--coordinate-tolerance",
        type=float,
        default=1e-3,
        help="Maximum world-to-crop center round-trip error in voxels.",
    )
    return parser.parse_args()


def load_uids(path: Path) -> list[str]:
    values = np.load(path, allow_pickle=False)
    return [str(value) for value in values]


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    args = parse_args()
    root = args.preprocessed_dir.expanduser().resolve()
    filelists = root / "filelists"
    annotation_path = root / "annotations_preprocessed.csv"
    if not root.is_dir():
        fail(f"Preprocessed directory does not exist: {root}")
    if not annotation_path.exists():
        fail(f"Missing annotation file: {annotation_path}")

    split_names = ("train", "val", "test")
    split_uids = {
        name: load_uids(filelists / f"{name}.npy") for name in split_names
    }
    split_sets = {name: set(values) for name, values in split_uids.items()}
    for name, values in split_uids.items():
        if len(values) != len(set(values)):
            fail(f"Duplicate seriesuid values in {name}.npy")
    for index, name in enumerate(split_names):
        for other in split_names[index + 1 :]:
            overlap = split_sets[name] & split_sets[other]
            if overlap:
                fail(f"{name} and {other} overlap on {len(overlap)} scans")

    all_uids = set().union(*split_sets.values())
    if len(all_uids) != args.expected_scans:
        fail(f"Expected {args.expected_scans} scans, found {len(all_uids)}")

    annotations = pd.read_csv(annotation_path, dtype={"pid": str})
    missing_columns = REQUIRED_COLUMNS - set(annotations.columns)
    if missing_columns:
        fail(f"Missing annotation columns: {sorted(missing_columns)}")
    if len(annotations) != args.expected_annotations:
        fail(
            f"Expected {args.expected_annotations} annotations, found {len(annotations)}"
        )
    unknown_pids = set(annotations["pid"]) - all_uids
    if unknown_pids:
        fail(f"Annotations reference {len(unknown_pids)} unknown scans")

    numeric_columns = sorted(REQUIRED_COLUMNS - {"pid"})
    numeric = annotations[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        fail("Annotation table contains NaN or infinite values")

    lower_columns = ["zmin", "ymin", "xmin"]
    upper_columns = ["zmax", "ymax", "xmax"]
    lower = annotations[lower_columns].to_numpy(dtype=np.float64)
    upper = annotations[upper_columns].to_numpy(dtype=np.float64)
    if np.any(upper <= lower):
        fail("One or more annotations have non-positive box sizes")

    shape_by_uid = {}
    maximum_roundtrip_error = 0.0
    outside_boxes = []
    for index, uid in enumerate(sorted(all_uids), 1):
        clean_path = root / f"{uid}_clean.npy"
        meta_path = root / f"{uid}_meta.npz"
        if not clean_path.exists() or not meta_path.exists():
            fail(f"Missing clean/meta file for {uid}")

        clean = np.load(clean_path, mmap_mode="r")
        if clean.ndim != 4 or clean.shape[0] != 1:
            fail(f"{clean_path.name} has invalid shape {clean.shape}")
        if clean.dtype != np.uint8:
            fail(f"{clean_path.name} must be uint8, got {clean.dtype}")
        shape = np.asarray(clean.shape[1:], dtype=np.int64)
        shape_by_uid[uid] = shape

        with np.load(meta_path) as meta:
            meta_shape = meta["crop_shape_zyx"].astype(np.int64)
            crop_start = meta["crop_start_zyx"].astype(np.float64)
            origin = meta["resampled_origin_xyz"].astype(np.float64)
            spacing = meta["resampled_spacing_xyz"].astype(np.float64)
            direction = meta["resampled_direction"].astype(np.float64).reshape(3, 3)
        if not np.array_equal(shape, meta_shape):
            fail(
                f"Shape mismatch for {uid}: clean={shape.tolist()}, "
                f"meta={meta_shape.tolist()}"
            )

        scan_annos = annotations[annotations["pid"] == uid]
        if len(scan_annos):
            scan_lower = scan_annos[lower_columns].to_numpy(dtype=np.float64)
            scan_upper = scan_annos[upper_columns].to_numpy(dtype=np.float64)
            if np.any(scan_upper <= 0) or np.any(scan_lower >= shape[None, :]):
                outside_boxes.append(uid)

            crop_centers_zyx = (scan_lower + scan_upper) / 2.0
            world_xyz = scan_annos[["coordX", "coordY", "coordZ"]].to_numpy(
                dtype=np.float64
            )
            full_index_xyz = (
                (np.linalg.inv(direction) @ (world_xyz - origin).T).T / spacing
            )
            expected_crop_centers = full_index_xyz[:, ::-1] - crop_start
            error = float(np.max(np.abs(crop_centers_zyx - expected_crop_centers)))
            maximum_roundtrip_error = max(maximum_roundtrip_error, error)

        if index % 100 == 0 or index == len(all_uids):
            print(f"Checked {index}/{len(all_uids)} scans", flush=True)

    if outside_boxes:
        fail(f"Annotations lie completely outside {len(outside_boxes)} cropped scans")
    if maximum_roundtrip_error > args.coordinate_tolerance:
        fail(
            f"Coordinate round-trip error {maximum_roundtrip_error:.6g} exceeds "
            f"tolerance {args.coordinate_tolerance}"
        )

    for name in split_names:
        split_annotations = pd.read_csv(
            filelists / f"{name}_annos.csv", dtype={"pid": str}
        )
        expected = annotations[annotations["pid"].isin(split_sets[name])]
        if len(split_annotations) != len(expected):
            fail(
                f"{name}_annos.csv has {len(split_annotations)} rows; "
                f"expected {len(expected)}"
            )

    split_info_path = filelists / "split.json"
    split_info = (
        json.loads(split_info_path.read_text(encoding="utf-8"))
        if split_info_path.exists()
        else {}
    )
    shapes = np.stack(list(shape_by_uid.values()))
    print("\nLUNA16 preprocessing validation passed")
    print(f"Scans:       {len(all_uids)}")
    print(f"Annotations: {len(annotations)}")
    print(
        "Splits:      "
        + ", ".join(f"{name}={len(split_uids[name])}" for name in split_names)
    )
    print(f"Fold config: {split_info}")
    print(f"Shape min:   {shapes.min(axis=0).tolist()} [z,y,x]")
    print(f"Shape max:   {shapes.max(axis=0).tolist()} [z,y,x]")
    print(f"Round-trip:  {maximum_roundtrip_error:.3g} voxels")


if __name__ == "__main__":
    main()
