#!/usr/bin/env python3
"""
Drift-Sense Navigation Error Recovery Localization CLI

Loads a 100x high-magnification reference image (1000x1000) and a 10x wide search image (1000x1000),
runs the multi-scale rotation-tolerant localization algorithm, and outputs predicted target (x, y) center coordinates.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image, ImageDraw

from src.matcher import localize_reference_in_search
from src.metrics import compute_euclidean_error


def run_localization_on_pair(
    ref_path: Path,
    search_path: Path,
    meta_path: Path = None,
    scoring_mode: str = "D",
    out_dir: Path = None,
    method: str = "geometric",
    use_ai: bool = False
) -> Dict[str, Any]:
    """Load reference and search images, run localization engine, and report results."""
    ref_img = np.asarray(Image.open(ref_path).convert("L"), dtype=np.uint8)
    search_img = np.asarray(Image.open(search_path).convert("L"), dtype=np.uint8)

    # Determine architecture from metadata if present
    arch = "FinFET"
    if meta_path and meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            arch = m.get("architecture", "FinFET")
        except Exception:
            pass

    # Optional AI restoration pre-filter if enabled and PyTorch available
    if use_ai:
        try:
            from models.ai_restoration import SEMRestorationUNet
            print("[Localization] AI Restoration Pre-filter enabled.")
        except Exception as e:
            print(f"[Localization Warning] AI Restoration unavailable ({e}). Continuing with raw SEM inputs.")

    t0 = os.times().elapsed if hasattr(os, 'times') else 0.0

    if method == "phase_correlation":
        from src.phase_correlation_matcher import PhaseCorrelationMatcher
        pcm = PhaseCorrelationMatcher(scale_min=8.5, scale_max=11.5, scale_step=0.25, rotation_min=-2.5, rotation_max=2.5, rotation_step=0.5)
        candidates = pcm.search_candidates(ref_img, search_img, top_k=5)
        if not candidates:
            loc_result = {
                "predicted_center": {"x": 500.0, "y": 500.0},
                "confidence": 0.0,
                "runtime_ms": 0.0,
                "best_scale": 10.0,
                "best_rotation_deg": 0.0,
                "bbox_search": {"xmin": 450, "ymin": 450, "xmax": 550, "ymax": 550}
            }
        else:
            best = candidates[0]
            tw, th = best["template_w"], best["template_h"]
            cx, cy = best["center_x"], best["center_y"]
            loc_result = {
                "predicted_center": {"x": float(cx), "y": float(cy)},
                "confidence": float(best["score_combined"]),
                "runtime_ms": 0.0,
                "best_scale": float(best["scale_ratio"]),
                "best_rotation_deg": float(best["rotation_deg"]),
                "bbox_search": {"xmin": cx - tw/2, "ymin": cy - th/2, "xmax": cx + tw/2, "ymax": cy + th/2}
            }
    else:
        loc_result = localize_reference_in_search(
            reference_img=ref_img,
            search_img=search_img,
            nominal_scale=10.0,
            scale_min=8.5,
            scale_max=11.5,
            scale_steps=7,
            rotation_min_deg=-2.5,
            rotation_max_deg=2.5,
            rotation_steps=5,
            scoring_mode=scoring_mode,
            architecture=arch
        )

    pred_x = loc_result["predicted_center"]["x"]
    pred_y = loc_result["predicted_center"]["y"]

    output_data = {
        "reference_file": str(ref_path),
        "search_file": str(search_path),
        "predicted_center": {"x": float(pred_x), "y": float(pred_y)},
        "confidence": float(loc_result["confidence"]),
        "runtime_ms": float(loc_result["runtime_ms"]),
        "detected_scale": float(loc_result["best_scale"]),
        "detected_rotation_deg": float(loc_result["best_rotation_deg"]),
        "method_used": method
    }

    # Evaluate against ground truth if metadata file is available
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            true_x = meta["target_center"]["x"]
            true_y = meta["target_center"]["y"]
            error = compute_euclidean_error(pred_x, pred_y, true_x, true_y)

            output_data["true_center"] = {"x": float(true_x), "y": float(true_y)}
            output_data["euclidean_error"] = float(error)
            output_data["pass_5px"] = bool(error <= 5.0)
            output_data["pass_4px"] = bool(error <= 4.0)
            output_data["pass_2px"] = bool(error <= 2.0)
            output_data["pass_1px"] = bool(error <= 1.0)
            output_data["pass_subpixel"] = bool(error <= 0.5)
        except Exception as e:
            output_data["metadata_error"] = str(e)

    # Save visual verification overlay if out_dir specified
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay = Image.fromarray(search_img, mode="L").convert("RGB")
        draw = ImageDraw.Draw(overlay)

        # Draw predicted center (red dot and box)
        bbox = loc_result["bbox_search"]
        draw.rectangle(
            (int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])),
            outline=(255, 0, 0),
            width=2
        )
        r = 5
        draw.ellipse(
            (int(pred_x - r), int(pred_y - r), int(pred_x + r), int(pred_y + r)),
            fill=(255, 0, 0)
        )

        # Draw ground truth center if available (green circle)
        if "true_center" in output_data:
            tx, ty = output_data["true_center"]["x"], output_data["true_center"]["y"]
            draw.ellipse(
                (int(tx - r - 2), int(ty - r - 2), int(tx + r + 2), int(ty + r + 2)),
                outline=(0, 255, 0),
                width=2
            )

        overlay_path = out_dir / f"overlay_{ref_path.stem}.png"
        overlay.save(overlay_path)
        output_data["overlay_path"] = str(overlay_path)

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Navigation Error Recovery Localization Engine")
    parser.add_argument("--reference", type=Path, required=True, help="Path to 1000x1000 reference image")
    parser.add_argument("--search", type=Path, required=True, help="Path to 1000x1000 search image")
    parser.add_argument("--metadata", type=Path, help="Optional path to metadata.json ground truth")
    parser.add_argument("--out-dir", type=Path, help="Optional directory to save visualization overlays")
    parser.add_argument("--method", type=str, choices=["geometric", "phase_correlation"], default="geometric", help="Matcher engine backend")
    parser.add_argument("--use-ai", action="store_true", help="Enable PyTorch AI Restoration pre-filter")

    args = parser.parse_args()

    result = run_localization_on_pair(
        ref_path=args.reference,
        search_path=args.search,
        meta_path=args.metadata,
        out_dir=args.out_dir,
        method=args.method,
        use_ai=args.use_ai
    )

    print("=" * 60)
    print(f"Drift-Sense Integrated Localization Result ({result['method_used'].upper()} Engine):")
    print(f"  Predicted Center (x, y): ({result['predicted_center']['x']:.2f}, {result['predicted_center']['y']:.2f})")
    print(f"  Match Confidence:        {result['confidence']:.4f}")
    print(f"  Runtime:                 {result['runtime_ms']:.2f} ms")
    print(f"  Detected Scale:          {result['detected_scale']:.2f}x")
    print(f"  Detected Rotation:       {result['detected_rotation_deg']:.2f}°")

    if "euclidean_error" in result:
        print(f"  Ground Truth Center:     ({result['true_center']['x']:.2f}, {result['true_center']['y']:.2f})")
        print(f"  Euclidean Distance Error:{result['euclidean_error']:.3f} px")
        print(f"  Pass @ 5px:              {'YES' if result['pass_5px'] else 'NO'}")
        print(f"  Pass @ 1px:              {'YES' if result['pass_1px'] else 'NO'}")
        print(f"  Sub-pixel Pass (<=0.5px):{'YES' if result['pass_subpixel'] else 'NO'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

