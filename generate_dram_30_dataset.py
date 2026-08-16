#!/usr/bin/env python3
"""
Drift-Sense DRAM 30-Pair Engineered Dataset Generator

Generates:
1. 30 deliberately engineered DRAM reference/search pairs across 6 difficulty tiers.
2. Ground-truth metadata and post-rotation coordinate labels.
3. 30-pair visual Contact Sheet (`dram_30_contact_sheet.png`) for visual ground-truth verification.
4. Packaged ML dataset in `DRAM_30/` (reference/, search/, labels.csv, dataset_manifest.json).
5. Unseen evaluation dataset in `DRAM_15_eval/` for final model validation.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generate_dataset import generate_single_pair, transform_point_rotation, transform_bbox_rotation


# Define the 30 case specifications across 6 difficulty categories
CASE_SPECS = [
    # 1-5: Easy (~10x, low noise, little/no rotation)
    {"tier": "Easy", "scale": 10.0, "rot": 0.0, "blur": 0.4, "dose": 250.0, "gstd": 3.0, "chg": 0.0, "tx": 500.0, "ty": 500.0},
    {"tier": "Easy", "scale": 10.0, "rot": 0.0, "blur": 0.4, "dose": 240.0, "gstd": 3.5, "chg": 0.0, "tx": 480.0, "ty": 520.0},
    {"tier": "Easy", "scale": 9.95, "rot": 0.1, "blur": 0.4, "dose": 260.0, "gstd": 3.0, "chg": 0.0, "tx": 530.0, "ty": 470.0},
    {"tier": "Easy", "scale": 10.05, "rot": -0.1, "blur": 0.5, "dose": 230.0, "gstd": 4.0, "chg": 0.0, "tx": 460.0, "ty": 540.0},
    {"tier": "Easy", "scale": 10.0, "rot": 0.0, "blur": 0.4, "dose": 250.0, "gstd": 3.0, "chg": 0.0, "tx": 510.0, "ty": 490.0},

    # 6-10: Scale Variation (9.0x - 11.0x)
    {"tier": "Scale", "scale": 9.1, "rot": 0.0, "blur": 0.4, "dose": 240.0, "gstd": 3.5, "chg": 0.0, "tx": 500.0, "ty": 500.0},
    {"tier": "Scale", "scale": 9.5, "rot": 0.0, "blur": 0.5, "dose": 250.0, "gstd": 4.0, "chg": 0.0, "tx": 470.0, "ty": 530.0},
    {"tier": "Scale", "scale": 10.0, "rot": 0.0, "blur": 0.4, "dose": 250.0, "gstd": 3.5, "chg": 0.0, "tx": 520.0, "ty": 480.0},
    {"tier": "Scale", "scale": 10.5, "rot": 0.0, "blur": 0.5, "dose": 230.0, "gstd": 4.0, "chg": 0.0, "tx": 450.0, "ty": 550.0},
    {"tier": "Scale", "scale": 10.9, "rot": 0.0, "blur": 0.5, "dose": 220.0, "gstd": 4.5, "chg": 0.0, "tx": 530.0, "ty": 460.0},

    # 11-15: Rotation Variation (+-1 deg to +-2 deg)
    {"tier": "Rotation", "scale": 10.0, "rot": -1.8, "blur": 0.4, "dose": 240.0, "gstd": 3.5, "chg": 0.0, "tx": 500.0, "ty": 500.0},
    {"tier": "Rotation", "scale": 10.0, "rot": -1.0, "blur": 0.5, "dose": 250.0, "gstd": 4.0, "chg": 0.0, "tx": 480.0, "ty": 520.0},
    {"tier": "Rotation", "scale": 10.0, "rot": +0.8, "blur": 0.4, "dose": 230.0, "gstd": 4.0, "chg": 0.0, "tx": 520.0, "ty": 480.0},
    {"tier": "Rotation", "scale": 10.0, "rot": +1.5, "blur": 0.5, "dose": 220.0, "gstd": 4.5, "chg": 0.0, "tx": 460.0, "ty": 540.0},
    {"tier": "Rotation", "scale": 10.0, "rot": +2.0, "blur": 0.5, "dose": 210.0, "gstd": 5.0, "chg": 0.0, "tx": 540.0, "ty": 460.0},

    # 16-20: SEM Degradation (blur + noise + charging)
    {"tier": "SEM Degradation", "scale": 10.0, "rot": 0.0, "blur": 0.8, "dose": 120.0, "gstd": 8.0, "chg": 10.0, "tx": 500.0, "ty": 500.0},
    {"tier": "SEM Degradation", "scale": 10.0, "rot": 0.0, "blur": 1.0, "dose": 100.0, "gstd": 10.0, "chg": 10.0, "tx": 470.0, "ty": 530.0},
    {"tier": "SEM Degradation", "scale": 10.0, "rot": 0.0, "blur": 1.1, "dose": 90.0, "gstd": 12.0, "chg": 15.0, "tx": 530.0, "ty": 470.0},
    {"tier": "SEM Degradation", "scale": 10.0, "rot": 0.0, "blur": 1.2, "dose": 80.0, "gstd": 14.0, "chg": 20.0, "tx": 450.0, "ty": 550.0},
    {"tier": "SEM Degradation", "scale": 10.0, "rot": 0.0, "blur": 1.3, "dose": 70.0, "gstd": 15.0, "chg": 20.0, "tx": 520.0, "ty": 480.0},

    # 21-25: Combined (scale + rotation + degradation)
    {"tier": "Combined", "scale": 9.2, "rot": +1.6, "blur": 0.6, "dose": 160.0, "gstd": 6.0, "chg": 10.0, "tx": 350.0, "ty": 650.0},
    {"tier": "Combined", "scale": 9.6, "rot": -1.4, "blur": 0.7, "dose": 140.0, "gstd": 7.0, "chg": 10.0, "tx": 650.0, "ty": 350.0},
    {"tier": "Combined", "scale": 10.4, "rot": +1.8, "blur": 0.8, "dose": 130.0, "gstd": 8.0, "chg": 10.0, "tx": 300.0, "ty": 700.0},
    {"tier": "Combined", "scale": 10.8, "rot": -1.7, "blur": 0.9, "dose": 110.0, "gstd": 10.0, "chg": 15.0, "tx": 700.0, "ty": 300.0},
    {"tier": "Combined", "scale": 9.4, "rot": +1.2, "blur": 0.8, "dose": 120.0, "gstd": 9.0, "chg": 10.0, "tx": 400.0, "ty": 600.0},

    # 26-30: Hard DRAM (highly periodic / ambiguous / edge location)
    {"tier": "Hard DRAM", "scale": 9.1, "rot": +1.9, "blur": 1.1, "dose": 80.0, "gstd": 12.0, "chg": 20.0, "tx": 220.0, "ty": 220.0},
    {"tier": "Hard DRAM", "scale": 10.9, "rot": -1.9, "blur": 1.2, "dose": 75.0, "gstd": 13.0, "chg": 20.0, "tx": 780.0, "ty": 780.0},
    {"tier": "Hard DRAM", "scale": 9.2, "rot": -2.0, "blur": 1.2, "dose": 70.0, "gstd": 14.0, "chg": 20.0, "tx": 250.0, "ty": 750.0},
    {"tier": "Hard DRAM", "scale": 10.8, "rot": +2.0, "blur": 1.3, "dose": 65.0, "gstd": 15.0, "chg": 25.0, "tx": 750.0, "ty": 250.0},
    {"tier": "Hard DRAM", "scale": 9.1, "rot": +1.9, "blur": 1.4, "dose": 60.0, "gstd": 16.0, "chg": 25.0, "tx": 200.0, "ty": 800.0},
]


def generate_dram_dataset(
    out_dir: Path,
    specs: List[Dict[str, Any]],
    seed_base: int = 5000
) -> List[Dict[str, Any]]:
    """Generate reference/search pairs based on exact specs."""
    out_dir = Path(out_dir)
    ref_dir = out_dir / "reference"
    search_dir = out_dir / "search"

    ref_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []

    for i, spec in enumerate(specs, start=1):
        ref_name = f"dram_{i:03d}_ref.png"
        search_name = f"dram_{i:03d}_search.png"
        meta_name = f"dram_{i:03d}_meta.json"

        seed = seed_base + i

        meta = generate_single_pair(
            out_dir=out_dir,
            pair_id=i,
            architecture="DRAM",
            target_x=spec["tx"],
            target_y=spec["ty"],
            scale_ratio=spec["scale"],
            rotation_deg=spec["rot"],
            edge_bloom=0.3,
            blur_sigma=spec["blur"],
            shot_dose=spec["dose"],
            gaussian_std=spec["gstd"],
            charging_strength=spec["chg"],
            contrast=1.0,
            gamma=1.0,
            seed=seed,
            ref_filename=f"reference/{ref_name}",
            search_filename=f"search/{search_name}",
            meta_filename=meta_name,
            save_preview=False
        )

        meta["tier"] = spec["tier"]
        metadata_list.append(meta)
        print(f"  [DRAM_{i:03d}] Tier={spec['tier']:<15} Scale={spec['scale']:4.1f}x Rot={spec['rot']:+4.1f}° Center=({meta['target_center']['x']:.1f}, {meta['target_center']['y']:.1f})")

    # Write labels.csv
    csv_path = out_dir / "labels.csv"
    fieldnames = [
        "pair_id", "reference", "search", "x", "y",
        "xmin", "ymin", "xmax", "ymax",
        "scale", "rotation", "blur", "shot_noise", "gaussian_noise", "charging", "seed"
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for meta in metadata_list:
            pid = meta["pair_id"]
            ref_rel = f"reference/dram_{pid:03d}_ref.png"
            search_rel = f"search/dram_{pid:03d}_search.png"
            tc = meta["target_center"]
            tb = meta["target_bbox"]
            sem = meta["sem_parameters"]

            writer.writerow({
                "pair_id": pid,
                "reference": ref_rel,
                "search": search_rel,
                "x": round(tc["x"], 4),
                "y": round(tc["y"], 4),
                "xmin": round(tb["xmin"], 4),
                "ymin": round(tb["ymin"], 4),
                "xmax": round(tb["xmax"], 4),
                "ymax": round(tb["ymax"], 4),
                "scale": round(meta["scale_ratio"], 4),
                "rotation": round(meta["rotation_deg"], 4),
                "blur": round(sem["blur_sigma"], 4),
                "shot_noise": round(sem["shot_dose"], 4),
                "gaussian_noise": round(sem["gaussian_std"], 4),
                "charging": round(sem["charging_strength"], 4),
                "seed": sem["seed"]
            })

    # Write dataset_manifest.json
    manifest = {
        "dataset_name": "Drift-Sense DRAM 30-Pair Engineered Dataset",
        "version": "v1.7_post_rotation_corrected",
        "architecture": "DRAM",
        "total_pairs": len(specs),
        "difficulty_tiers": {
            "Easy": "Cases 1-5: ~10x scale, 0° rotation, clean low noise",
            "Scale": "Cases 6-10: Scale range 9.1x - 10.9x",
            "Rotation": "Cases 11-15: Rotation range ±1.0° to ±2.0°",
            "SEM Degradation": "Cases 16-20: Heavy blur, low dose, gaussian noise, charging",
            "Combined": "Cases 21-25: Simultaneous scale, rotation, degradation, offset center",
            "Hard DRAM": "Cases 26-30: Edge placement, periodic ambiguity, severe noise & charging"
        },
        "labels_file": "labels.csv"
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSaved packaged dataset to {out_dir} (Manifest: labels.csv, dataset_manifest.json)")
    return metadata_list


def create_contact_sheet(
    dataset_dir: Path,
    metadata_list: List[Dict[str, Any]],
    output_path: Path
):
    """
    Generate a 30-pair Contact Sheet (5x6 grid) showing reference and search images with ground truth annotations.
    """
    dataset_dir = Path(dataset_dir)
    rows = 6
    cols = 5
    num_pairs = len(metadata_list)

    # Individual thumbnail width: 360px (170 ref + 170 search + 20 pad)
    # Individual thumbnail height: 210px (170 image + 40 label)
    cell_w = 360
    cell_h = 210
    pad = 10
    header_h = 60

    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = header_h + rows * cell_h + (rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(24, 28, 36))
    draw = ImageDraw.Draw(canvas)

    # Draw header text
    draw.text((canvas_w // 2 - 250, 15), "DRIFT-SENSE DRAM 30-PAIR DATASET CONTACT SHEET", fill=(255, 255, 255))
    draw.text((canvas_w // 2 - 220, 35), "Reference (100x FOV) vs Search (10x FOV) with Ground-Truth Coordinates", fill=(180, 200, 220))

    for idx, meta in enumerate(metadata_list):
        r = idx // cols
        c = idx % cols

        x0 = pad + c * (cell_w + pad)
        y0 = header_h + pad + r * (cell_h + pad)

        pid = meta["pair_id"]
        tier = meta.get("tier", "DRAM")
        ref_path = dataset_dir / f"reference/dram_{pid:03d}_ref.png"
        search_path = dataset_dir / f"search/dram_{pid:03d}_search.png"

        if not ref_path.exists() or not search_path.exists():
            continue

        ref_img = Image.open(ref_path).convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
        search_img = Image.open(search_path).convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)

        # Draw ground truth center marker on reference crop thumbnail (center = 80, 80)
        draw_ref = ImageDraw.Draw(ref_img)
        draw_ref.ellipse([75, 75, 85, 85], outline=(0, 255, 0), width=2)
        draw_ref.line([80, 70, 80, 90], fill=(0, 255, 0), width=1)
        draw_ref.line([70, 80, 90, 80], fill=(0, 255, 0), width=1)

        # Draw ground truth center and bbox on search crop thumbnail
        draw_search = ImageDraw.Draw(search_img)
        tc = meta["target_center"]
        tb = meta["target_bbox"]
        sx = (tc["x"] / 1000.0) * 160.0
        sy = (tc["y"] / 1000.0) * 160.0
        bx0 = (tb["xmin"] / 1000.0) * 160.0
        by0 = (tb["ymin"] / 1000.0) * 160.0
        bx1 = (tb["xmax"] / 1000.0) * 160.0
        by1 = (tb["ymax"] / 1000.0) * 160.0

        draw_search.rectangle([bx0, by0, bx1, by1], outline=(0, 255, 0), width=2)
        draw_search.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=(255, 0, 0), outline=(255, 255, 255))

        # Paste ref and search thumbnails into cell
        canvas.paste(ref_img, (x0 + 10, y0 + 5))
        canvas.paste(search_img, (x0 + 180, y0 + 5))

        # Draw cell border and text metadata
        draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(60, 70, 90), width=1)

        scale = meta["scale_ratio"]
        rot = meta["rotation_deg"]
        text_label = f"#{pid:02d} [{tier}] s={scale:.1f}x r={rot:+.1f}°"
        text_coords = f"GT: ({tc['x']:.0f}, {tc['y']:.0f})"

        draw.text((x0 + 10, y0 + 172), text_label, fill=(220, 230, 245))
        draw.text((x0 + 180, y0 + 172), text_coords, fill=(160, 220, 160))

    canvas.save(output_path)
    print(f"Generated 30-pair Contact Sheet: {output_path}")


def main():
    base_dir = Path(r"D:\SemiconFINFETwork")
    dram_30_dir = base_dir / "DRAM_30"
    dram_eval_dir = base_dir / "DRAM_15_eval"
    contact_sheet_path = base_dir / "dram_30_contact_sheet.png"

    print("=" * 75)
    print("STEP 1 & 2: Generating 30 Deliberately Engineered DRAM Pairs...")
    print("=" * 75)
    meta_30 = generate_dram_dataset(dram_30_dir, CASE_SPECS, seed_base=5000)

    print("\n" + "=" * 75)
    print("STEP 3: Creating 30-Pair Visual Inspection Contact Sheet...")
    print("=" * 75)
    create_contact_sheet(dram_30_dir, meta_30, contact_sheet_path)

    print("\n" + "=" * 75)
    print("Generating 15 Unseen DRAM Evaluation Pairs (DRAM_15_eval)...")
    print("=" * 75)
    eval_specs = CASE_SPECS[::2]  # Sample 15 distinct cases across tiers
    generate_dram_dataset(dram_eval_dir, eval_specs, seed_base=6000)

    print("\nGeneration Complete!")


if __name__ == "__main__":
    main()
