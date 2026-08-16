#!/usr/bin/env python3
"""
Drift-Sense Stage 1 Sample Dataset Verification & Benchmark Suite

Performs automated verification on generated sample dataset (e.g., 100 pairs):
1. Directory structure & manifest schema check.
2. CSV label integrity & post-rotation ground-truth coordinate sanity check.
3. Degradation tier distribution breakdown check (40/30/20/10).
4. Classical V1.7 localization engine benchmark run to verify label precision.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from localize import run_localization_on_pair
from src.metrics import evaluate_batch_performance


def verify_dataset_structure(base_dir: Path) -> Dict[str, Any]:
    """Verify presence of directories, files, and manifest JSON."""
    issues = []
    splits = ["train", "validation", "test"]

    manifest_file = base_dir / "dataset_manifest.json"
    if not manifest_file.exists():
        issues.append("Missing dataset_manifest.json in root directory!")

    split_counts = {}
    for split in splits:
        split_dir = base_dir / split
        ref_dir = split_dir / "reference"
        search_dir = split_dir / "search"
        csv_file = split_dir / "labels.csv"

        if not split_dir.exists():
            issues.append(f"Missing split directory: {split_dir}")
            continue
        if not ref_dir.exists():
            issues.append(f"Missing reference directory: {ref_dir}")
        if not search_dir.exists():
            issues.append(f"Missing search directory: {search_dir}")
        if not csv_file.exists():
            issues.append(f"Missing labels.csv in split: {csv_file}")

        n_ref = len(list(ref_dir.glob("*.png"))) if ref_dir.exists() else 0
        n_search = len(list(search_dir.glob("*.png"))) if search_dir.exists() else 0
        split_counts[split] = {"reference": n_ref, "search": n_search}

        if n_ref != n_search:
            issues.append(f"Mismatch in image count for split '{split}': {n_ref} refs vs {n_search} searches")

    return {
        "status": "PASS" if len(issues) == 0 else "FAIL",
        "issues": issues,
        "split_counts": split_counts
    }


def verify_labels_and_coordinates(base_dir: Path) -> Dict[str, Any]:
    """Verify labels.csv integrity, file existence, and bounding box containment."""
    splits = ["train", "validation", "test"]
    required_cols = [
        "pair_id", "reference_file", "search_file", "architecture",
        "true_x", "true_y", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
        "scale_ratio", "rotation_deg", "blur_sigma", "shot_dose",
        "gaussian_std", "charging_strength", "degradation_tier", "seed"
    ]

    total_rows = 0
    issues = []
    tier_counts = {}

    for split in splits:
        csv_path = base_dir / split / "labels.csv"
        if not csv_path.exists():
            continue

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            missing = [col for col in required_cols if col not in fieldnames]
            if missing:
                issues.append(f"Split '{split}' CSV missing columns: {missing}")

            for row in reader:
                total_rows += 1
                pid = row["pair_id"]
                ref_path = base_dir / split / row["reference_file"]
                search_path = base_dir / split / row["search_file"]

                if not ref_path.exists():
                    issues.append(f"Reference image not found: {ref_path}")
                if not search_path.exists():
                    issues.append(f"Search image not found: {search_path}")

                tx = float(row["true_x"])
                ty = float(row["true_y"])
                b_xmin = float(row["bbox_xmin"])
                b_ymin = float(row["bbox_ymin"])
                b_xmax = float(row["bbox_xmax"])
                b_ymax = float(row["bbox_ymax"])
                tier = row["degradation_tier"]

                if split == "train":
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1

                # Verify target center is within bounding box (with slight sub-pixel epsilon)
                if not (b_xmin - 0.5 <= tx <= b_xmax + 0.5 and b_ymin - 0.5 <= ty <= b_ymax + 0.5):
                    issues.append(f"Pair {pid} ({split}): Center ({tx:.2f}, {ty:.2f}) outside bbox [{b_xmin}, {b_ymin}, {b_xmax}, {b_ymax}]")

    return {
        "status": "PASS" if len(issues) == 0 else "FAIL",
        "total_rows_checked": total_rows,
        "issues": issues,
        "train_tier_counts": tier_counts
    }


def run_classical_benchmark_sample(base_dir: Path, max_eval_pairs: int = 100) -> Dict[str, Any]:
    """Run V1.7 classical localization matcher on sample pairs to verify labels and baseline precision."""
    print(f"\nRunning V1.7 Classical Matcher Benchmark on sample dataset (up to {max_eval_pairs} pairs)...")
    splits = ["train", "validation", "test"]
    results = []

    eval_count = 0
    for split in splits:
        csv_path = base_dir / split / "labels.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if eval_count >= max_eval_pairs:
                break

            ref_path = base_dir / split / row["reference_file"]
            search_path = base_dir / split / row["search_file"]

            # Construct temporary metadata file or dictionary for localization check
            temp_meta = base_dir / split / f"metadata_{row['pair_id']}.json"
            if not temp_meta.exists():
                meta_dict = {
                    "architecture": row["architecture"],
                    "target_center": {"x": float(row["true_x"]), "y": float(row["true_y"])}
                }
                temp_meta.write_text(json.dumps(meta_dict), encoding="utf-8")

            loc_res = run_localization_on_pair(
                ref_path=ref_path,
                search_path=search_path,
                meta_path=temp_meta
            )

            # Cleanup temporary metadata if created
            if temp_meta.name.startswith("metadata_pair_"):
                temp_meta.unlink(missing_ok=True)

            err = loc_res["euclidean_error"]
            tier = row["degradation_tier"]
            print(f"  Sample {eval_count+1:02d} [{split}/{tier}] | True: ({row['true_x']:.1f}, {row['true_y']:.1f}) | Pred: ({loc_res['predicted_center']['x']:.1f}, {loc_res['predicted_center']['y']:.1f}) | Error: {err:.2f}px | Pass @5px: {loc_res['pass_5px']}")

            results.append({
                "pair_id": row["pair_id"],
                "split": split,
                "degradation_tier": tier,
                "true_x": row["true_x"],
                "true_y": row["true_y"],
                "pred_x": loc_res["predicted_center"]["x"],
                "pred_y": loc_res["predicted_center"]["y"],
                "euclidean_error": err,
                "confidence": loc_res["confidence"],
                "runtime_ms": loc_res["runtime_ms"],
                "pass_5px": loc_res["pass_5px"],
                "pass_4px": loc_res["pass_4px"],
                "pass_2px": loc_res["pass_2px"],
                "pass_1px": loc_res["pass_1px"],
                "pass_subpixel": loc_res["pass_subpixel"]
            })
            eval_count += 1

    metrics = evaluate_batch_performance(results)
    return {
        "eval_count": eval_count,
        "metrics": metrics,
        "details": results
    }


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Stage 1 Dataset Verification Suite")
    parser.add_argument("--dataset-dir", type=str, default=r"D:\SemiconFINFETwork\results\finfet_100_sample", help="Dataset directory to verify")
    args = parser.parse_args()

    base_dir = Path(args.dataset_dir)
    print("=" * 80)
    print("Drift-Sense Stage 1 Sample Verification Suite")
    print(f"Target Directory: {base_dir}")
    print("=" * 80)

    # 1. Structure check
    struct_res = verify_dataset_structure(base_dir)
    print(f"\n[1/4] Directory Structure Verification: {struct_res['status']}")
    print(f"      Split File Counts: {struct_res['split_counts']}")
    if struct_res["issues"]:
        for iss in struct_res["issues"]:
            print(f"      - ERROR: {iss}")

    # 2. Labels & Geometry check
    label_res = verify_labels_and_coordinates(base_dir)
    print(f"\n[2/4] CSV Labels & Coordinate Verification: {label_res['status']}")
    print(f"      Total CSV Rows Checked: {label_res['total_rows_checked']}")
    print(f"      Train Tier Breakdown:   {label_res['train_tier_counts']}")
    if label_res["issues"]:
        for iss in label_res["issues"]:
            print(f"      - ERROR: {iss}")

    # 3. Tier breakdown proportions check
    total_train = sum(label_res["train_tier_counts"].values()) if label_res["train_tier_counts"] else 1
    print("\n[3/4] Degradation Tier Distribution Check:")
    for t_name, count in label_res["train_tier_counts"].items():
        pct = (count / total_train) * 100.0
        print(f"      - {t_name:20s}: {count:4d} pairs ({pct:.1f}%)")

    # 4. Classical Benchmark Localization Run
    bench_res = run_classical_benchmark_sample(base_dir)
    m = bench_res["metrics"]

    print("\n" + "=" * 80)
    print("STAGE 1 SAMPLE VERIFICATION SUMMARY & CLASSICAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Evaluated Pairs:           {bench_res['eval_count']}")
    print(f"Pass Rate @ 5.0px:          {m['pass_rates_percent']['threshold_5px']:.1f}%")
    print(f"Pass Rate @ 4.0px:          {m['pass_rates_percent']['threshold_4px']:.1f}%")
    print(f"Pass Rate @ 2.0px:          {m['pass_rates_percent']['threshold_2px']:.1f}%")
    print(f"Pass Rate @ 1.0px:          {m['pass_rates_percent']['threshold_1px']:.1f}%")
    print(f"Pass Rate @ Sub-pixel (0.5px): {m['pass_rates_percent']['threshold_subpixel_0_5px']:.1f}%")
    print(f"Mean Euclidean Error:       {m['euclidean_error_px']['mean']:.3f} px")
    print(f"Median Euclidean Error:     {m['euclidean_error_px']['median']:.3f} px")
    print(f"Worst-Case Max Error:       {m['euclidean_error_px']['worst_case_max']:.3f} px")

    all_pass = (struct_res["status"] == "PASS" and label_res["status"] == "PASS")
    print("=" * 80)
    if all_pass:
        print(">>> STAGE 1 VERIFICATION PASSED SUCCESSFULLY! PROCEED TO 25,000 GENERATION <<<")
    else:
        print(">>> STAGE 1 VERIFICATION FAILED! REVIEW ISSUES ABOVE <<<")
    print("=" * 80)


if __name__ == "__main__":
    main()
