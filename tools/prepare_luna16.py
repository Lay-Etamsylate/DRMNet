#!/usr/bin/env python3
"""Prepare the official LUNA16 MHD scans for DRMNet.

The script performs the missing repository preprocessing step:

1. resample CT and the supplied lung mask to isotropic spacing;
2. crop to a dilated lung bounding box;
3. window HU values to uint8 and save ``<seriesuid>_clean.npy``;
4. convert world-coordinate LUNA annotations to crop voxel boxes;
5. generate fold, train, validation, and test file lists.

All saved arrays use NumPy/SimpleITK order ``[z, y, x]``. Physical points use
the LUNA convention ``[x, y, z]``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


HU_MIN = -1200.0
HU_MAX = 600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/home/zpc/jc/data/LUNA16"),
        help="Directory containing subset0..subset9, annotations.csv and lung masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-root>/drmnet_preprocessed).",
    )
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument(
        "--lung-margin-mm",
        type=float,
        default=10.0,
        help="Dilation/crop margin around the supplied lung mask.",
    )
    parser.add_argument("--pad-value", type=int, default=170)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--test-fold", type=int, default=0, choices=range(10))
    parser.add_argument("--val-fold", type=int, default=1, choices=range(10))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only-splits",
        action="store_true",
        help="Regenerate file lists from already preprocessed scans.",
    )
    return parser.parse_args()


def discover_scans(data_root: Path) -> list[tuple[str, int, Path]]:
    scans = []
    for fold in range(10):
        subset_dir = data_root / f"subset{fold}"
        for path in sorted(subset_dir.glob("*.mhd")):
            scans.append((path.stem, fold, path))
    if not scans:
        raise FileNotFoundError(f"No subset0..subset9 MHD scans found below {data_root}")
    duplicates = pd.Series([uid for uid, _, _ in scans]).duplicated()
    if duplicates.any():
        raise RuntimeError("Duplicate seriesuid values were found in LUNA subsets")
    return scans


def discover_masks(data_root: Path) -> dict[str, Path]:
    mask_root = data_root / "seg-lungs-LUNA16"
    masks = {path.stem: path for path in mask_root.rglob("*.mhd")}
    if not masks:
        raise FileNotFoundError(f"No lung masks found below {mask_root}")
    return masks


def resample_image(
    image: sitk.Image,
    spacing_xyz: tuple[float, float, float],
    interpolator: int,
    default_value: float,
) -> sitk.Image:
    old_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    old_size = np.asarray(image.GetSize(), dtype=np.int64)
    new_spacing = np.asarray(spacing_xyz, dtype=np.float64)
    new_size = np.maximum(1, np.rint(old_size * old_spacing / new_spacing)).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(tuple(float(v) for v in new_spacing))
    resampler.SetSize([int(v) for v in new_size])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(image)


def align_mask(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask)


def hu_to_uint8(image: np.ndarray) -> np.ndarray:
    scaled = (image.astype(np.float32) - HU_MIN) / (HU_MAX - HU_MIN)
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def annotation_boxes(
    annotations: list[dict],
    resampled: sitk.Image,
    crop_start_zyx: np.ndarray,
) -> list[dict]:
    rows = []
    spacing_xyz = np.asarray(resampled.GetSpacing(), dtype=np.float64)
    spacing_zyx = spacing_xyz[::-1]
    for annotation in annotations:
        world_xyz = (
            float(annotation["coordX"]),
            float(annotation["coordY"]),
            float(annotation["coordZ"]),
        )
        index_xyz = np.asarray(
            resampled.TransformPhysicalPointToContinuousIndex(world_xyz),
            dtype=np.float64,
        )
        center_zyx = index_xyz[::-1] - crop_start_zyx
        diameter_mm = float(annotation["diameter_mm"])
        size_zyx = diameter_mm / spacing_zyx
        lower = center_zyx - size_zyx / 2.0
        upper = center_zyx + size_zyx / 2.0
        rows.append(
            {
                "pid": annotation["seriesuid"],
                "zmin": lower[0],
                "zmax": upper[0],
                "ymin": lower[1],
                "ymax": upper[1],
                "xmin": lower[2],
                "xmax": upper[2],
                "coordX": world_xyz[0],
                "coordY": world_xyz[1],
                "coordZ": world_xyz[2],
                "diameter_mm": diameter_mm,
            }
        )
    return rows


def process_one(task: dict) -> dict:
    uid = task["uid"]
    output_dir = Path(task["output_dir"])
    clean_path = output_dir / f"{uid}_clean.npy"
    meta_path = output_dir / f"{uid}_meta.npz"
    if clean_path.exists() and meta_path.exists() and not task["overwrite"]:
        with np.load(meta_path) as meta:
            rows = json.loads(str(meta["annotation_rows_json"].item()))
            shape = tuple(int(v) for v in meta["crop_shape_zyx"])
        return {"uid": uid, "status": "skipped", "rows": rows, "shape": shape}

    ct = sitk.ReadImage(task["scan_path"])
    mask = sitk.ReadImage(task["mask_path"])
    spacing = float(task["spacing"])
    ct_resampled = resample_image(
        ct, (spacing, spacing, spacing), sitk.sitkLinear, HU_MIN
    )
    mask_resampled = align_mask(mask, ct_resampled)

    ct_array = sitk.GetArrayFromImage(ct_resampled).astype(np.float32, copy=False)
    lung_mask = sitk.GetArrayFromImage(mask_resampled) > 0
    if not lung_mask.any():
        raise RuntimeError(f"Lung mask is empty for {uid}")

    iterations = max(0, int(math.ceil(float(task["margin_mm"]) / spacing)))
    if iterations:
        lung_mask = ndimage.binary_dilation(lung_mask, iterations=iterations)

    coordinates = np.where(lung_mask)
    crop_start = np.asarray([axis.min() for axis in coordinates], dtype=np.int64)
    crop_end = np.asarray([axis.max() + 1 for axis in coordinates], dtype=np.int64)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(crop_start, crop_end))

    clean = hu_to_uint8(ct_array[slices])
    crop_mask = lung_mask[slices]
    clean[~crop_mask] = np.uint8(task["pad_value"])
    clean = clean[np.newaxis, ...]

    rows = annotation_boxes(task["annotations"], ct_resampled, crop_start)
    atomic_save_npy(clean_path, clean)
    atomic_save_npz(
        meta_path,
        seriesuid=np.asarray(uid),
        fold=np.asarray(task["fold"], dtype=np.int16),
        crop_start_zyx=crop_start,
        crop_shape_zyx=np.asarray(clean.shape[1:], dtype=np.int64),
        resampled_origin_xyz=np.asarray(ct_resampled.GetOrigin(), dtype=np.float64),
        resampled_spacing_xyz=np.asarray(ct_resampled.GetSpacing(), dtype=np.float64),
        resampled_direction=np.asarray(ct_resampled.GetDirection(), dtype=np.float64),
        source_origin_xyz=np.asarray(ct.GetOrigin(), dtype=np.float64),
        source_spacing_xyz=np.asarray(ct.GetSpacing(), dtype=np.float64),
        source_direction=np.asarray(ct.GetDirection(), dtype=np.float64),
        annotation_rows_json=np.asarray(json.dumps(rows)),
    )
    return {
        "uid": uid,
        "status": "processed",
        "rows": rows,
        "shape": tuple(int(v) for v in clean.shape[1:]),
    }


def save_split(
    name: str,
    uids: Iterable[str],
    annotations: pd.DataFrame,
    filelist_dir: Path,
) -> None:
    uids = np.asarray(sorted(uids), dtype=str)
    np.save(filelist_dir / f"{name}.npy", uids)
    subset = annotations[annotations["pid"].isin(set(uids))].copy()
    subset.to_csv(filelist_dir / f"{name}_annos.csv", index=False)


def generate_splits(
    scans: list[tuple[str, int, Path]],
    annotations: pd.DataFrame,
    output_dir: Path,
    test_fold: int,
    val_fold: int,
) -> None:
    if test_fold == val_fold:
        raise ValueError("--test-fold and --val-fold must differ")
    filelist_dir = output_dir / "filelists"
    filelist_dir.mkdir(parents=True, exist_ok=True)
    fold_to_uids = {
        fold: [uid for uid, scan_fold, _ in scans if scan_fold == fold]
        for fold in range(10)
    }
    for fold, uids in fold_to_uids.items():
        save_split(f"fold{fold}", uids, annotations, filelist_dir)

    test_uids = fold_to_uids[test_fold]
    val_uids = fold_to_uids[val_fold]
    train_uids = [
        uid
        for fold, uids in fold_to_uids.items()
        if fold not in {test_fold, val_fold}
        for uid in uids
    ]
    save_split("train", train_uids, annotations, filelist_dir)
    save_split("val", val_uids, annotations, filelist_dir)
    save_split("test", test_uids, annotations, filelist_dir)
    # The official LUNA evaluator expects this file without a header.
    pd.DataFrame({"seriesuid": sorted(test_uids)}).to_csv(
        filelist_dir / "test_seriesuids.csv", index=False, header=False
    )
    split_info = {
        "test_fold": test_fold,
        "val_fold": val_fold,
        "train_folds": [f for f in range(10) if f not in {test_fold, val_fold}],
        "counts": {
            "train": len(train_uids),
            "val": len(val_uids),
            "test": len(test_uids),
        },
    }
    (filelist_dir / "split.json").write_text(
        json.dumps(split_info, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else data_root / "drmnet_preprocessed"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scans = discover_scans(data_root)
    source_annotations = pd.read_csv(data_root / "annotations.csv")
    annotations_by_uid = {
        str(uid): group.to_dict("records")
        for uid, group in source_annotations.groupby("seriesuid")
    }

    processed_rows = []
    if args.only_splits:
        annotation_path = output_dir / "annotations_preprocessed.csv"
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)
        processed_annotations = pd.read_csv(annotation_path)
    else:
        masks = discover_masks(data_root)
        missing_masks = sorted(uid for uid, _, _ in scans if uid not in masks)
        if missing_masks:
            preview = ", ".join(missing_masks[:5])
            raise FileNotFoundError(
                f"Missing lung masks for {len(missing_masks)} scans, including: {preview}"
            )
        tasks = [
            {
                "uid": uid,
                "fold": fold,
                "scan_path": str(scan_path),
                "mask_path": str(masks[uid]),
                "output_dir": str(output_dir),
                "spacing": args.spacing,
                "margin_mm": args.lung_margin_mm,
                "pad_value": args.pad_value,
                "overwrite": args.overwrite,
                "annotations": annotations_by_uid.get(uid, []),
            }
            for uid, fold, scan_path in scans
        ]
        completed = 0
        shape_rows = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, task): task["uid"] for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed += 1
                processed_rows.extend(result["rows"])
                shape_rows.append(
                    {"seriesuid": result["uid"], "shape_zyx": result["shape"]}
                )
                print(
                    f"[{completed:03d}/{len(tasks):03d}] "
                    f"{result['status']:9s} {result['uid']} {result['shape']}",
                    flush=True,
                )
        columns = [
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
        ]
        processed_annotations = pd.DataFrame(processed_rows, columns=columns)
        processed_annotations.to_csv(
            output_dir / "annotations_preprocessed.csv", index=False
        )
        pd.DataFrame(shape_rows).sort_values("seriesuid").to_csv(
            output_dir / "scan_shapes.csv", index=False
        )

    generate_splits(
        scans, processed_annotations, output_dir, args.test_fold, args.val_fold
    )
    print(f"Prepared data: {output_dir}")
    print(f"Scans: {len(scans)}; annotations: {len(processed_annotations)}")
    print(f"Split files: {output_dir / 'filelists'}")


if __name__ == "__main__":
    main()
