#!/usr/bin/env python3
"""
Drift-Sense DRAM Benchmark Suite with PyTorch ML Model (SEMRestorationUNet)
=============================================================================
Executes fast, comprehensive DRAM wafer-only evaluation comparing:
  1. Classical CV Pipeline (src.matcher.localize_reference_in_search)
  2. PyTorch ML-Restored Pipeline (SEMRestorationUNet U-Net Pre-filter + src.matcher)

Evaluates on:
  - 30 Synthetic DRAM SEM Image Pairs (data/generated/)
  - 30 Engineered DRAM Benchmark Test Pairs (DRAM_30/)

Outputs:
  - Summary JSON: results/metrics/dram_ml_benchmark_summary.json
  - CSV Table:    results/metrics/dram_ml_benchmark_per_sample.csv
"""

import sys
import os
import json
import time
import math
import csv
from pathlib import Path

BASE_DIR = Path(r"D:\SemiconFINFETwork")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import torch

from models.ai_restoration import SEMRestorationUNet
from src.matcher import localize_reference_in_search
from src.metrics import compute_euclidean_error


def preprocess_with_unet(img: np.ndarray, model: SEMRestorationUNet) -> np.ndarray:
    """Pre-process SEM image using PyTorch SEMRestorationUNet model."""
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        out_tensor = model(img_tensor)
    restored = (out_tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return restored


def run_dram_ml_benchmark():
    print("=" * 110, flush=True)
    print("  DRIFT-SENSE  |  DRAM BENCHMARK WITH IMPORTED ML MODEL (SEMRestorationUNet)", flush=True)
    print("  PyTorch Version: 2.8.0+cpu  |  Target Architecture: DRAM High-Density Array", flush=True)
    print("  ML Pre-filter  : SEMRestorationUNet Edge-Preserving U-Net", flush=True)
    print("=" * 110, flush=True)
    print(flush=True)

    # Initialize PyTorch ML Model
    ml_model = SEMRestorationUNet(in_channels=1, out_channels=1)
    ml_model.eval()
    print("[ML Engine] PyTorch SEMRestorationUNet loaded successfully.\n", flush=True)

    ds1_results = []
    ds2_results = []

    # ── DATASET 1: data/generated (30 DRAM SEM pairs) ──────────────────────────
    gen_dir = BASE_DIR / "data" / "generated"
    manifest_path = BASE_DIR / "data" / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        print("-" * 110, flush=True)
        print(f"EVALUATING DATASET 1: {len(manifest)} DRAM Synthetic Pairs (data/generated/)", flush=True)
        print(f"{'ID':>3} | {'Scale':>5} | {'Rot':>5} | {'True (X,Y)':>14} | {'CV Pred (X,Y)':>15} | {'CV Err':>8} | {'ML+CV Pred':>15} | {'ML Err':>8} | {'Status'}", flush=True)
        print("-" * 110, flush=True)

        for meta in manifest:
            sid = meta["sample_id"]
            tx, ty = float(meta["true_x"]), float(meta["true_y"])
            scale, rot = float(meta["scale_ratio"]), float(meta["rotation_deg"])

            ref_path = gen_dir / f"sample_{sid:03d}_ref.png"
            srch_path = gen_dir / f"sample_{sid:03d}_search.png"

            if not ref_path.exists() or not srch_path.exists():
                continue

            ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            srch_img = cv2.imread(str(srch_path), cv2.IMREAD_GRAYSCALE)
            if ref_img is None or srch_img is None:
                continue

            # 1. Classical CV Pipeline
            t0_cv = time.perf_counter()
            res_cv = localize_reference_in_search(
                ref_img, srch_img, nominal_scale=scale,
                scale_min=8.5, scale_max=11.5, scale_steps=5,
                rotation_min_deg=-2.0, rotation_max_deg=2.0, rotation_steps=3,
                scoring_mode="D", architecture="DRAM"
            )
            t1_cv = time.perf_counter()
            cv_ms = (t1_cv - t0_cv) * 1000.0
            px_cv = res_cv["predicted_center"]["x"]
            py_cv = res_cv["predicted_center"]["y"]
            cv_err = compute_euclidean_error(px_cv, py_cv, tx, ty)

            # 2. PyTorch ML Restoration + CV Pipeline
            t0_ml = time.perf_counter()
            ref_ml = preprocess_with_unet(ref_img, ml_model)
            srch_ml = preprocess_with_unet(srch_img, ml_model)
            res_ml = localize_reference_in_search(
                ref_ml, srch_ml, nominal_scale=scale,
                scale_min=8.5, scale_max=11.5, scale_steps=5,
                rotation_min_deg=-2.0, rotation_max_deg=2.0, rotation_steps=3,
                scoring_mode="D", architecture="DRAM"
            )
            t1_ml = time.perf_counter()
            ml_ms = (t1_ml - t0_ml) * 1000.0
            px_ml = res_ml["predicted_center"]["x"]
            py_ml = res_ml["predicted_center"]["y"]
            ml_err = compute_euclidean_error(px_ml, py_ml, tx, ty)

            status_cv = "✓" if cv_err <= 5.0 else "✗"
            status_ml = "✓" if ml_err <= 5.0 else "✗"

            print(f"{sid:02d} | {scale:5.1f} | {rot:+5.1f} | ({tx:5.1f},{ty:5.1f}) | ({px_cv:5.1f},{py_cv:5.1f}) | {cv_err:6.2f}px | ({px_ml:5.1f},{py_ml:5.1f}) | {ml_err:6.2f}px | CV:{status_cv} ML:{status_ml}", flush=True)

            ds1_results.append({
                "sample_id": sid,
                "dataset": "generated_dram",
                "scale_ratio": scale,
                "rotation_deg": rot,
                "true_x": tx,
                "true_y": ty,
                "cv_pred_x": px_cv,
                "cv_pred_y": py_cv,
                "cv_error_px": cv_err,
                "cv_runtime_ms": cv_ms,
                "cv_pass_5px": cv_err <= 5.0,
                "ml_pred_x": px_ml,
                "ml_pred_y": py_ml,
                "ml_error_px": ml_err,
                "ml_runtime_ms": ml_ms,
                "ml_pass_5px": ml_err <= 5.0
            })

    # ── DATASET 2: DRAM_30/ (30 engineered DRAM pairs) ─────────────────────────
    dram30_dir = BASE_DIR / "DRAM_30"
    meta_files = sorted(dram30_dir.glob("dram_*_meta.json"))

    if meta_files:
        print("\n" + "-" * 110, flush=True)
        print(f"EVALUATING DATASET 2: {len(meta_files)} Engineered DRAM Test Pairs (DRAM_30/)", flush=True)
        print(f"{'Pair':>4} | {'Tier':<16} | {'True (X,Y)':>14} | {'CV Pred (X,Y)':>15} | {'CV Err':>8} | {'ML+CV Pred':>15} | {'ML Err':>8} | {'Status'}", flush=True)
        print("-" * 110, flush=True)

        for meta_path in meta_files:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pid  = meta["pair_id"]
            tier = meta.get("tier", "DRAM")
            tx   = float(meta["target_center"]["x"])
            ty   = float(meta["target_center"]["y"])

            ref_path = dram30_dir / meta["reference"]["filename"]
            srch_path = dram30_dir / meta["search"]["filename"]

            if not ref_path.exists() or not srch_path.exists():
                continue

            ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            srch_img = cv2.imread(str(srch_path), cv2.IMREAD_GRAYSCALE)
            if ref_img is None or srch_img is None:
                continue

            # 1. Classical CV
            t0_cv = time.perf_counter()
            res_cv = localize_reference_in_search(
                ref_img, srch_img, nominal_scale=10.0,
                scale_min=8.5, scale_max=11.5, scale_steps=7,
                rotation_min_deg=-2.5, rotation_max_deg=2.5, rotation_steps=5,
                scoring_mode="D", architecture="DRAM"
            )
            t1_cv = time.perf_counter()
            cv_ms = (t1_cv - t0_cv) * 1000.0
            px_cv = res_cv["predicted_center"]["x"]
            py_cv = res_cv["predicted_center"]["y"]
            cv_err = compute_euclidean_error(px_cv, py_cv, tx, ty)

            # 2. PyTorch ML Model + CV
            t0_ml = time.perf_counter()
            ref_ml = preprocess_with_unet(ref_img, ml_model)
            srch_ml = preprocess_with_unet(srch_img, ml_model)
            res_ml = localize_reference_in_search(
                ref_ml, srch_ml, nominal_scale=10.0,
                scale_min=8.5, scale_max=11.5, scale_steps=7,
                rotation_min_deg=-2.5, rotation_max_deg=2.5, rotation_steps=5,
                scoring_mode="D", architecture="DRAM"
            )
            t1_ml = time.perf_counter()
            ml_ms = (t1_ml - t0_ml) * 1000.0
            px_ml = res_ml["predicted_center"]["x"]
            py_ml = res_ml["predicted_center"]["y"]
            ml_err = compute_euclidean_error(px_ml, py_ml, tx, ty)

            status_cv = "✓" if cv_err <= 5.0 else "✗"
            status_ml = "✓" if ml_err <= 5.0 else "✗"

            print(f"{pid:02d}   | {tier:<16} | ({tx:5.1f},{ty:5.1f}) | ({px_cv:5.1f},{py_cv:5.1f}) | {cv_err:6.2f}px | ({px_ml:5.1f},{py_ml:5.1f}) | {ml_err:6.2f}px | CV:{status_cv} ML:{status_ml}", flush=True)

            ds2_results.append({
                "pair_id": pid,
                "dataset": "engineered_dram30",
                "tier": tier,
                "scale_ratio": meta["scale_ratio"],
                "rotation_deg": meta["rotation_deg"],
                "true_x": tx,
                "true_y": ty,
                "cv_pred_x": px_cv,
                "cv_pred_y": py_cv,
                "cv_error_px": cv_err,
                "cv_runtime_ms": cv_ms,
                "cv_pass_5px": cv_err <= 5.0,
                "ml_pred_x": px_ml,
                "ml_pred_y": py_ml,
                "ml_error_px": ml_err,
                "ml_runtime_ms": ml_ms,
                "ml_pass_5px": ml_err <= 5.0
            })

    # ── SUMMARY PERFORMANCE METRICS ───────────────────────────────────────────
    print("\n" + "=" * 110, flush=True)
    print("FINAL DRAM BENCHMARK SUMMARY (CV vs PyTorch SEMRestorationUNet ML Model)", flush=True)
    print("=" * 110, flush=True)

    if ds1_results:
        cv_e1 = [r["cv_error_px"] for r in ds1_results]
        ml_e1 = [r["ml_error_px"] for r in ds1_results]
        cv_p1 = sum(1 for r in ds1_results if r["cv_pass_5px"])
        ml_p1 = sum(1 for r in ds1_results if r["ml_pass_5px"])
        N1 = len(ds1_results)

        print(f"DATASET 1: Generated DRAM (N={N1})", flush=True)
        print(f"  - Classical CV Pipeline : Pass Rate = {cv_p1}/{N1} ({cv_p1/N1*100:.1f}%) | Mean Error = {np.mean(cv_e1):.2f} px | Median = {np.median(cv_e1):.2f} px", flush=True)
        print(f"  - PyTorch ML Pipeline   : Pass Rate = {ml_p1}/{N1} ({ml_p1/N1*100:.1f}%) | Mean Error = {np.mean(ml_e1):.2f} px | Median = {np.median(ml_e1):.2f} px", flush=True)

    if ds2_results:
        cv_e2 = [r["cv_error_px"] for r in ds2_results]
        ml_e2 = [r["ml_error_px"] for r in ds2_results]
        cv_p2 = sum(1 for r in ds2_results if r["cv_pass_5px"])
        ml_p2 = sum(1 for r in ds2_results if r["ml_pass_5px"])
        N2 = len(ds2_results)

        print(f"\nDATASET 2: Engineered DRAM_30 Benchmark (N={N2})", flush=True)
        print(f"  - Classical CV Pipeline : Pass Rate = {cv_p2}/{N2} ({cv_p2/N2*100:.1f}%) | Mean Error = {np.mean(cv_e2):.2f} px | Median = {np.median(cv_e2):.2f} px", flush=True)
        print(f"  - PyTorch ML Pipeline   : Pass Rate = {ml_p2}/{N2} ({ml_p2/N2*100:.1f}%) | Mean Error = {np.mean(ml_e2):.2f} px | Median = {np.median(ml_e2):.2f} px", flush=True)

    print("=" * 110, flush=True)

    # ── SAVE SUMMARY JSON & CSV ───────────────────────────────────────────────
    out_dir = BASE_DIR / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = {
        "target_wafer": "DRAM Memory Array",
        "ml_model_architecture": "SEMRestorationUNet (Edge-Preserving U-Net PyTorch 2.8.0+cpu)",
        "dataset_1_generated": {
            "total_pairs": len(ds1_results),
            "classical_cv": {
                "pass_rate_5px": sum(1 for r in ds1_results if r["cv_pass_5px"]) / len(ds1_results) * 100.0 if ds1_results else 0.0,
                "mean_error_px": float(np.mean([r["cv_error_px"] for r in ds1_results])) if ds1_results else 0.0,
                "median_error_px": float(np.median([r["cv_error_px"] for r in ds1_results])) if ds1_results else 0.0,
            },
            "pytorch_ml": {
                "pass_rate_5px": sum(1 for r in ds1_results if r["ml_pass_5px"]) / len(ds1_results) * 100.0 if ds1_results else 0.0,
                "mean_error_px": float(np.mean([r["ml_error_px"] for r in ds1_results])) if ds1_results else 0.0,
                "median_error_px": float(np.median([r["ml_error_px"] for r in ds1_results])) if ds1_results else 0.0,
            },
            "per_pair_results": ds1_results
        },
        "dataset_2_engineered": {
            "total_pairs": len(ds2_results),
            "classical_cv": {
                "pass_rate_5px": sum(1 for r in ds2_results if r["cv_pass_5px"]) / len(ds2_results) * 100.0 if ds2_results else 0.0,
                "mean_error_px": float(np.mean([r["cv_error_px"] for r in ds2_results])) if ds2_results else 0.0,
                "median_error_px": float(np.median([r["cv_error_px"] for r in ds2_results])) if ds2_results else 0.0,
            },
            "pytorch_ml": {
                "pass_rate_5px": sum(1 for r in ds2_results if r["ml_pass_5px"]) / len(ds2_results) * 100.0 if ds2_results else 0.0,
                "mean_error_px": float(np.mean([r["ml_error_px"] for r in ds2_results])) if ds2_results else 0.0,
                "median_error_px": float(np.median([r["ml_error_px"] for r in ds2_results])) if ds2_results else 0.0,
            },
            "per_pair_results": ds2_results
        }
    }

    json_path = out_dir / "dram_ml_benchmark_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    csv_path = out_dir / "dram_ml_benchmark_per_sample.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset", "pair_id", "tier", "scale_ratio", "rotation_deg", "true_x", "true_y",
                      "cv_pred_x", "cv_pred_y", "cv_error_px", "cv_runtime_ms", "cv_pass_5px",
                      "ml_pred_x", "ml_pred_y", "ml_error_px", "ml_runtime_ms", "ml_pass_5px"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ds1_results)
        writer.writerows(ds2_results)

    print(f"\n[OK] Benchmark output saved to:\n  - JSON: {json_path}\n  - CSV:  {csv_path}", flush=True)


if __name__ == "__main__":
    run_dram_ml_benchmark()
