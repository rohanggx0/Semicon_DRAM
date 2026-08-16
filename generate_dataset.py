#!/usr/bin/env python3
"""
Drift-Sense Synthetic SEM Dataset Generator — V1.7 Post-Rotation Coordinate Transformation

Generates realistic reference/search SEM image pairs for FinFET & DRAM layouts.

Post-Rotation Coordinate Transformation:
- Search image relative rotation rotates the physical image pixels by rotation_deg around (499.5, 499.5).
- Target coordinates (x_true, y_true) and bounding boxes (xmin, ymin, xmax, ymax) are transformed
  by 2D rotation matrix R(theta) to maintain 100% mathematical ground-truth accuracy.
"""

import argparse
import math
from pathlib import Path
import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw

from src.geometry import generate_scene_and_pair
from src.sem_effects import apply_full_sem_pipeline


def transform_point_rotation(
    x: float, y: float,
    angle_deg: float,
    cx: float = 499.5, cy: float = 499.5
) -> Tuple[float, float]:
    """
    Rotate point (x, y) around image center (cx, cy) by angle_deg degrees
    matching PIL.Image.rotate(angle_deg, resample=Image.Resampling.BILINEAR, expand=False).
    """
    if abs(angle_deg) <= 1e-4:
        return float(x), float(y)

    rad = math.radians(-angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    dx = x - cx
    dy = y - cy

    x_rot = cx + dx * cos_a - dy * sin_a
    y_rot = cy + dx * sin_a + dy * cos_a

    return float(np.clip(x_rot, 0.0, 1000.0)), float(np.clip(y_rot, 0.0, 1000.0))


def transform_bbox_rotation(
    xmin: float, ymin: float, xmax: float, ymax: float,
    angle_deg: float,
    cx: float = 499.5, cy: float = 499.5
) -> Dict[str, float]:
    """Rotate bounding box corners and compute new axis-aligned bounding box."""
    if abs(angle_deg) <= 1e-4:
        return {"xmin": float(xmin), "ymin": float(ymin), "xmax": float(xmax), "ymax": float(ymax)}

    corners = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax)
    ]
    rot_corners = [transform_point_rotation(x, y, angle_deg, cx, cy) for x, y in corners]
    xs = [c[0] for c in rot_corners]
    ys = [c[1] for c in rot_corners]

    return {
        "xmin": float(np.clip(min(xs), 0.0, 1000.0)),
        "ymin": float(np.clip(min(ys), 0.0, 1000.0)),
        "xmax": float(np.clip(max(xs), 0.0, 1000.0)),
        "ymax": float(np.clip(max(ys), 0.0, 1000.0))
    }


def generate_single_pair(
    out_dir: Path,
    pair_id: int = 1,
    architecture: str = "FinFET",
    target_x: float = 500.0,
    target_y: float = 500.0,
    scale_ratio: float = 10.0,
    rotation_deg: float = 0.0,
    edge_bloom: float = 0.3,
    blur_sigma: float = 0.5,
    shot_dose: float = 200.0,
    gaussian_std: float = 5.0,
    charging_strength: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    seed: int = 42,
    ref_filename: Optional[str] = None,
    search_filename: Optional[str] = None,
    meta_filename: Optional[str] = None,
    preview_filename: Optional[str] = None,
    save_preview: bool = True
) -> Dict[str, Any]:
    """Generate one synthetic image pair with corrected post-rotation ground-truth metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base clean geometry generation
    raw_ref, raw_search, mapping_meta = generate_scene_and_pair(
        architecture=architecture,
        target_cx_search=target_x,
        target_cy_search=target_y,
        scale_ratio=scale_ratio,
        ref_size=1000,
        search_size=1000,
        variant_seed=seed
    )

    # 2. Apply SEM transfer functions and degradations independently
    ref_sem = apply_full_sem_pipeline(
        raw_ref,
        edge_bloom=edge_bloom,
        blur_sigma=blur_sigma,
        shot_dose=shot_dose * 1.5,
        gaussian_std=gaussian_std * 0.5,
        charging_strength=0.0,
        contrast=contrast,
        gamma=gamma,
        rotation_deg=0.0,  # Reference orientation fixed at 0.0 degrees
        seed=seed
    )

    search_sem = apply_full_sem_pipeline(
        raw_search,
        edge_bloom=edge_bloom,
        blur_sigma=blur_sigma * 1.2,
        shot_dose=shot_dose,
        gaussian_std=gaussian_std,
        charging_strength=charging_strength,
        contrast=contrast,
        gamma=gamma,
        rotation_deg=rotation_deg,  # Genuine relative rotation vs reference
        seed=seed + 1000
    )

    # 3. Save images
    ref_name = ref_filename if ref_filename else f"reference_{pair_id:03d}.png"
    search_name = search_filename if search_filename else f"search_{pair_id:03d}.png"
    meta_name = meta_filename if meta_filename else f"metadata_{pair_id:03d}.json"
    preview_name = preview_filename if preview_filename else f"preview_{pair_id:03d}.png"

    ref_path = out_dir / ref_name
    search_path = out_dir / search_name
    meta_path = out_dir / meta_name
    preview_path = out_dir / preview_name

    ref_path.parent.mkdir(parents=True, exist_ok=True)
    search_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(ref_sem, mode="L").save(ref_path)
    Image.fromarray(search_sem, mode="L").save(search_path)

    # 4. Transform ground-truth center and bbox to reflect search image rotation
    rot_tx, rot_ty = transform_point_rotation(target_x, target_y, rotation_deg)
    raw_bbox = mapping_meta["target_bbox"]
    rot_bbox = transform_bbox_rotation(raw_bbox["xmin"], raw_bbox["ymin"], raw_bbox["xmax"], raw_bbox["ymax"], rotation_deg)

    mapping_meta["target_center"] = {"x": float(rot_tx), "y": float(rot_ty)}
    mapping_meta["target_bbox"] = rot_bbox
    mapping_meta["unrotated_target_center"] = {"x": float(target_x), "y": float(target_y)}

    # 5. Construct complete metadata
    metadata = {
        "pair_id": pair_id,
        "architecture": architecture,
        "version": "v1.7_post_rotation_corrected",
        "image_size_px": [1000, 1000],
        "scale_ratio": float(scale_ratio),
        "rotation_deg": float(rotation_deg),
        "reference": {
            "filename": ref_name,
            "fov_relative": 1.0
        },
        "search": {
            "filename": search_name,
            "fov_relative": float(scale_ratio)
        },
        "target_center": mapping_meta["target_center"],
        "target_bbox": mapping_meta["target_bbox"],
        "unrotated_target_center": mapping_meta["unrotated_target_center"],
        "sem_parameters": {
            "edge_bloom": float(edge_bloom),
            "blur_sigma": float(blur_sigma),
            "shot_dose": float(shot_dose),
            "gaussian_std": float(gaussian_std),
            "charging_strength": float(charging_strength),
            "contrast": float(contrast),
            "gamma": float(gamma),
            "seed": seed
        }
    }

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 6. Save visual side-by-side preview if requested
    if save_preview:
        canvas = Image.new("L", (2050, 1000), color=0)
        canvas.paste(Image.fromarray(ref_sem), (0, 0))
        canvas.paste(Image.fromarray(search_sem), (1050, 0))

        draw = ImageDraw.Draw(canvas)

        # Reference center marker
        draw.ellipse([490, 490, 510, 510], outline=255, width=2)
        draw.line([500, 480, 500, 520], fill=255, width=2)
        draw.line([480, 500, 520, 500], fill=255, width=2)

        # Search ground truth box (shifted by 1050px horizontal offset)
        b_xmin = rot_bbox["xmin"] + 1050
        b_ymin = rot_bbox["ymin"]
        b_xmax = rot_bbox["xmax"] + 1050
        b_ymax = rot_bbox["ymax"]
        draw.rectangle([b_xmin, b_ymin, b_xmax, b_ymax], outline=255, width=3)

        # Search target center marker
        scx = rot_tx + 1050
        scy = rot_ty
        draw.ellipse([scx - 8, scy - 8, scx + 8, scy + 8], outline=255, width=2)

        canvas.save(preview_path)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Dataset Generator V1.7")
    parser.add_argument("--architecture", type=str, default="FinFET", choices=["FinFET", "DRAM"], help="Layout architecture")
    parser.add_argument("--num-pairs", type=int, default=5, help="Number of synthetic image pairs to generate")
    parser.add_argument("--out-dir", type=str, default=r"D:\SemiconFINFETwork\results\synthetic_dataset", help="Output directory")
    args = parser.parse_args()

    num_pairs = args.num_pairs if hasattr(args, "num_pairs") else getattr(args, "num-pairs", 5)
    out_dir = Path(args.out_dir)

    print(f"Generating {num_pairs} {args.architecture} dataset pair(s) into {out_dir}...")
    rng = np.random.default_rng(2026)

    for i in range(1, num_pairs + 1):
        tx = float(rng.uniform(250.0, 750.0))
        ty = float(rng.uniform(250.0, 750.0))
        scale = float(rng.uniform(9.1, 10.9))
        rot = float(rng.uniform(-2.0, 2.0))
        blur = float(rng.uniform(0.4, 1.0))
        dose = float(rng.uniform(120.0, 280.0))
        gdev = float(rng.uniform(4.0, 10.0))
        chg = float(rng.choice([0.0, 10.0, 20.0]))

        meta = generate_single_pair(
            out_dir=out_dir,
            pair_id=i,
            architecture=args.architecture,
            target_x=tx,
            target_y=ty,
            scale_ratio=scale,
            rotation_deg=rot,
            edge_bloom=0.3,
            blur_sigma=blur,
            shot_dose=dose,
            gaussian_std=gdev,
            charging_strength=chg,
            seed=100 + i
        )
        print(f"  [{i}/{num_pairs}] Pair generated: Corrected True Center = ({meta['target_center']['x']:.1f}, {meta['target_center']['y']:.1f})")

    print(f"Dataset generation complete! Files saved to {out_dir}")


if __name__ == "__main__":
    main()
