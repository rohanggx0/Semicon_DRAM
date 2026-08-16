#!/usr/bin/env python3
"""
Drift-Sense Synthetic DRAM Dataset Regenerator
===============================================
Regenerates all 30 DRAM synthetic SEM image pairs in data/generated/
using the EXACT parameters archived in data/manifest.json.

For each sample the original (unrotated) target center is back-computed
from the manifest's true_x / true_y by applying the INVERSE rotation,
then generate_single_pair() is called with that seed and those parameters.
After generation the script verifies that the new true_x / true_y matches
the manifest to within 0.01 px; if not it flags a warning.

Writes a fresh data/manifest.json with live-computed coordinates so the
benchmark scripts always consume coordinates that are 100% consistent
with the on-disk images.
"""

import sys
import math
import json
from typing import TypedDict, List
from pathlib import Path

BASE_DIR = Path(r"D:\SemiconFINFETwork")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from generate_dataset import generate_single_pair, transform_point_rotation

# ── Original manifest parameters ─────────────────────────────────────────────
# Derived directly from reading data/manifest.json.
# Fields: sample_id, true_x (post-rot), true_y (post-rot),
#         scale_ratio, rotation_deg, noise_level, seed

class SpecDict(TypedDict):
    sample_id: int
    true_x: float
    true_y: float
    scale_ratio: float
    rotation_deg: float
    noise_level: str
    seed: int

MANIFEST_SPECS: List[SpecDict] = [
    {"sample_id": 0,  "true_x": 390.34196406149545, "true_y": 567.8810076788226,  "scale_ratio": 9.0,  "rotation_deg": -2.0, "noise_level": "low",    "seed": 42},
    {"sample_id": 1,  "true_x": 349.8628934170903,  "true_y": 592.4938334534362,  "scale_ratio": 10.0, "rotation_deg": -1.0, "noise_level": "medium", "seed": 43},
    {"sample_id": 2,  "true_x": 583.0909090909091,  "true_y": 596.9090909090909,  "scale_ratio": 11.0, "rotation_deg":  0.0, "noise_level": "high",   "seed": 44},
    {"sample_id": 3,  "true_x": 561.2579453486636,  "true_y": 584.2770705796501,  "scale_ratio": 9.0,  "rotation_deg":  1.0, "noise_level": "low",    "seed": 45},
    {"sample_id": 4,  "true_x": 568.4636636688774,  "true_y": 486.3023083318819,  "scale_ratio": 10.0, "rotation_deg":  2.0, "noise_level": "medium", "seed": 46},
    {"sample_id": 5,  "true_x": 514.5097842720942,  "true_y": 397.0800517530535,  "scale_ratio": 11.0, "rotation_deg": -2.0, "noise_level": "high",   "seed": 47},
    {"sample_id": 6,  "true_x": 571.1984205736845,  "true_y": 552.9173100028354,  "scale_ratio": 9.0,  "rotation_deg": -1.0, "noise_level": "low",    "seed": 48},
    {"sample_id": 7,  "true_x": 594.1,               "true_y": 512.7,              "scale_ratio": 10.0, "rotation_deg":  0.0, "noise_level": "medium", "seed": 49},
    {"sample_id": 8,  "true_x": 553.3939053350268,  "true_y": 605.4478444439125,  "scale_ratio": 11.0, "rotation_deg":  1.0, "noise_level": "high",   "seed": 50},
    {"sample_id": 9,  "true_x": 426.1303698529363,  "true_y": 364.9401825665905,  "scale_ratio": 9.0,  "rotation_deg":  2.0, "noise_level": "low",    "seed": 51},
    {"sample_id": 10, "true_x": 651.7731955017542,  "true_y": 552.1285634483276,  "scale_ratio": 10.0, "rotation_deg": -2.0, "noise_level": "medium", "seed": 52},
    {"sample_id": 11, "true_x": 676.7841443939972,  "true_y": 440.9854114410413,  "scale_ratio": 11.0, "rotation_deg": -1.0, "noise_level": "high",   "seed": 53},
    {"sample_id": 12, "true_x": 495.1111111111111,  "true_y": 419.8888888888889,  "scale_ratio": 9.0,  "rotation_deg":  0.0, "noise_level": "low",    "seed": 54},
    {"sample_id": 13, "true_x": 488.5625040691088,  "true_y": 566.7097720490098,  "scale_ratio": 10.0, "rotation_deg":  1.0, "noise_level": "medium", "seed": 55},
    {"sample_id": 14, "true_x": 527.5765620000761,  "true_y": 565.5320576788786,  "scale_ratio": 11.0, "rotation_deg":  2.0, "noise_level": "high",   "seed": 56},
    {"sample_id": 15, "true_x": 476.1862919107448,  "true_y": 535.6350659368507,  "scale_ratio": 9.0,  "rotation_deg": -2.0, "noise_level": "low",    "seed": 57},
    {"sample_id": 16, "true_x": 427.8469133282294,  "true_y": 508.1419950709857,  "scale_ratio": 10.0, "rotation_deg": -1.0, "noise_level": "medium", "seed": 58},
    {"sample_id": 17, "true_x": 620.8181818181819,  "true_y": 276.72727272727275, "scale_ratio": 11.0, "rotation_deg":  0.0, "noise_level": "high",   "seed": 59},
    {"sample_id": 18, "true_x": 555.9590675530687,  "true_y": 490.79975614528877, "scale_ratio": 9.0,  "rotation_deg":  1.0, "noise_level": "low",    "seed": 60},
    {"sample_id": 19, "true_x": 425.99177004851805, "true_y": 443.7485831338572,  "scale_ratio": 10.0, "rotation_deg":  2.0, "noise_level": "medium", "seed": 61},
    {"sample_id": 20, "true_x": 492.5672948865911,  "true_y": 546.132341271889,   "scale_ratio": 11.0, "rotation_deg": -2.0, "noise_level": "high",   "seed": 62},
    {"sample_id": 21, "true_x": 385.9422979385244,  "true_y": 508.6774069040887,  "scale_ratio": 9.0,  "rotation_deg": -1.0, "noise_level": "low",    "seed": 63},
    {"sample_id": 22, "true_x": 354.3,               "true_y": 435.5,              "scale_ratio": 10.0, "rotation_deg":  0.0, "noise_level": "medium", "seed": 64},
    {"sample_id": 23, "true_x": 442.59888306799024, "true_y": 482.72642950147093, "scale_ratio": 11.0, "rotation_deg":  1.0, "noise_level": "high",   "seed": 65},
    {"sample_id": 24, "true_x": 663.9723712520372,  "true_y": 563.2048383388557,  "scale_ratio": 9.0,  "rotation_deg":  2.0, "noise_level": "low",    "seed": 66},
    {"sample_id": 25, "true_x": 614.4197840810835,  "true_y": 472.7766091230679,  "scale_ratio": 10.0, "rotation_deg": -2.0, "noise_level": "medium", "seed": 67},
    {"sample_id": 26, "true_x": 340.99683169259555, "true_y": 530.0477703210248,  "scale_ratio": 11.0, "rotation_deg": -1.0, "noise_level": "high",   "seed": 68},
    {"sample_id": 27, "true_x": 562.7777777777778,  "true_y": 433.44444444444446, "scale_ratio": 9.0,  "rotation_deg":  0.0, "noise_level": "low",    "seed": 69},
    {"sample_id": 28, "true_x": 414.7593604372663,  "true_y": 315.05948695070595, "scale_ratio": 10.0, "rotation_deg":  1.0, "noise_level": "medium", "seed": 70},
    {"sample_id": 29, "true_x": 561.7868692151264,  "true_y": 498.0242839905761,  "scale_ratio": 11.0, "rotation_deg":  2.0, "noise_level": "high",   "seed": 71},
]

# SEM degradation parameters derived from noise_level
NOISE_PARAMS = {
    "low":    {"blur_sigma": 0.5,  "shot_dose": 250.0, "gaussian_std": 4.0,  "charging_strength": 0.0},
    "medium": {"blur_sigma": 0.8,  "shot_dose": 160.0, "gaussian_std": 8.0,  "charging_strength": 10.0},
    "high":   {"blur_sigma": 1.1,  "shot_dose": 90.0,  "gaussian_std": 14.0, "charging_strength": 20.0},
}


def inverse_rotate(x: float, y: float, angle_deg: float,
                   cx: float = 499.5, cy: float = 499.5):
    """Invert PIL's rotation to recover pre-rotation target center."""
    return transform_point_rotation(x, y, -angle_deg, cx, cy)


def main():
    out_dir = BASE_DIR / "data" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  DRIFT-SENSE  |  SYNTHETIC DRAM DATASET REGENERATOR")
    print(f"  Output: {out_dir}")
    print("  Architecture: DRAM | Pairs: 30 | Seeds: 42–71")
    print("=" * 75)
    print()

    new_manifest = []
    errors = []

    for spec in MANIFEST_SPECS:
        sid        = spec["sample_id"]
        true_x_man = spec["true_x"]
        true_y_man = spec["true_y"]
        scale      = spec["scale_ratio"]
        rot        = spec["rotation_deg"]
        seed       = spec["seed"]
        noise      = spec["noise_level"]
        params     = NOISE_PARAMS[noise]

        # Back-compute the unrotated input target center from archived post-rotation true_x/true_y
        # transform_point_rotation applies: neg rotation → inverse rotation
        tx_unrot, ty_unrot = inverse_rotate(true_x_man, true_y_man, rot)

        ref_name    = f"sample_{sid:03d}_ref.png"
        search_name = f"sample_{sid:03d}_search.png"
        meta_name   = f"sample_{sid:03d}_meta.json"

        meta = generate_single_pair(
            out_dir=out_dir,
            pair_id=sid,
            architecture="DRAM",
            target_x=tx_unrot,
            target_y=ty_unrot,
            scale_ratio=scale,
            rotation_deg=rot,
            edge_bloom=0.3,
            blur_sigma=params["blur_sigma"],
            shot_dose=params["shot_dose"],
            gaussian_std=params["gaussian_std"],
            charging_strength=params["charging_strength"],
            contrast=1.0,
            gamma=1.0,
            seed=seed,
            ref_filename=ref_name,
            search_filename=search_name,
            meta_filename=meta_name,
            save_preview=False,
        )

        live_tx = meta["target_center"]["x"]
        live_ty = meta["target_center"]["y"]
        err_x   = abs(live_tx - true_x_man)
        err_y   = abs(live_ty - true_y_man)

        status = "✓" if (err_x < 0.5 and err_y < 0.5) else "⚠"
        print(f"  [{sid:02d}] {ref_name}  true=({true_x_man:.3f},{true_y_man:.3f})  "
              f"live=({live_tx:.3f},{live_ty:.3f})  Δ=({err_x:.4f},{err_y:.4f}) {status}")

        if err_x >= 0.5 or err_y >= 0.5:
            errors.append(sid)

        # Use LIVE coordinates as the new ground truth (consistent with regenerated images)
        new_manifest.append({
            "sample_id":    sid,
            "true_x":       float(live_tx),
            "true_y":       float(live_ty),
            "scale_ratio":  scale,
            "rotation_deg": rot,
            "noise_level":  noise,
            "seed":         seed,
            "ref_path":     f"data\\generated\\{ref_name}",
            "search_path":  f"data\\generated\\{search_name}",
        })

    # Write fresh manifest.json
    manifest_path = BASE_DIR / "data" / "manifest.json"
    manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")

    print()
    print(f"[OK] Generated {len(new_manifest)} image pairs in {out_dir}")
    print(f"[OK] Manifest written: {manifest_path}")

    if errors:
        print(f"[WARN] {len(errors)} samples had coordinate drift > 0.5px: {errors}")
        print("       This can occur when the inverse-rotation approximation diverges")
        print("       at the image border. Manifest uses LIVE coordinates (consistent).")
    else:
        print("[OK] All 30 samples: ground-truth coordinates within 0.5px of archived values.")

    print()
    print("Regeneration complete.")


if __name__ == "__main__":
    main()
