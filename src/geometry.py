"""
Synthetic Semiconductor Geometry Engine (FinFET & DRAM) — V1.3 Asymmetric Target Landmark & Scale Semantics

Key Innovation in V1.3:
-----------------------
- Breaks periodic correlation symmetry by introducing asymmetric local landmark geometry.
- All target-local features fit strictly inside the 1000x1000 HR reference crop (+-500 HR px).
- Resolves the 11:1 scale periodic ambiguity where symmetric features produced competing correlation peaks.

Physical Scale Semantics:
- Reference Image: Fixed 1000x1000 high-magnification field (1x FOV)
- High-Resolution Scene: (search_size * scale_ratio) x (search_size * scale_ratio) physical extent
- Search Image: Downsampled scene to 1000x1000 pixels (scale_ratio FOV)
"""

from typing import Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image


def draw_finfet_scene(
    size: int = 10000,
    target_cx_hr: int = 5000,
    target_cy_hr: int = 5000,
    fin_pitch_hr: float = 90.0,
    fin_width_hr: float = 24.0,
    gate_width_hr: float = 30.0
) -> np.ndarray:
    """
    Generate FinFET high-resolution continuous scene with V1.3 Asymmetric Target Landmarks.

    V1.3 Asymmetric Target Features:
    - Upper parallel gate bar (dy = -180 HR px): 320 HR px width (-160 to +160 HR px)
    - Lower parallel gate bar (dy = +180 HR px): 200 HR px width, right-shifted (-60 to +140 HR px)
    - Contact pad: 30x30 HR px, offset to (+35, -20) HR px from target center
    - Local fin modification: one fin at +90 HR px has a local STI bridge widening (36 HR px width)
    """
    h = w = size
    img = np.zeros((h, w), dtype=np.uint8)

    # 1. Vertical silicon fins (175 intensity)
    x = 0.0
    while x < w + fin_pitch_hr:
        x0 = max(0, int(round(x - fin_width_hr / 2.0)))
        x1 = min(w, int(round(x + fin_width_hr / 2.0)))
        if x1 > x0:
            img[:, x0:x1] = 175
        x += fin_pitch_hr

    # 2. Local fin symmetry breaker: widen one fin near target (+90 HR px) for a 120 HR px vertical stretch
    special_fin_x = target_cx_hr + int(round(fin_pitch_hr))
    sf_x0 = max(0, special_fin_x - 18)
    sf_x1 = min(w, special_fin_x + 18)
    sf_y0 = max(0, target_cy_hr - 60)
    sf_y1 = min(h, target_cy_hr + 60)
    if sf_x1 > sf_x0 and sf_y1 > sf_y0:
        img[sf_y0:sf_y1, sf_x0:sf_x1] = 200

    # 3. Continuous horizontal main gate (245 intensity)
    main_gate_y = target_cy_hr
    y0 = max(0, int(round(main_gate_y - gate_width_hr / 2.0)))
    y1 = min(h, int(round(main_gate_y + gate_width_hr / 2.0)))
    if y1 > y0:
        img[y0:y1, :] = 245

    # 4. Asymmetric target-local parallel gate bars (255 intensity)
    local_gw = int(round(gate_width_hr))

    # Upper gate bar (dy = -180 HR px): width 320 HR px (-160 to +160)
    gy_up = target_cy_hr - 180
    uy0, uy1 = max(0, gy_up - local_gw // 2), min(h, gy_up + local_gw // 2)
    ux0, ux1 = max(0, target_cx_hr - 160), min(w, target_cx_hr + 160)
    if uy1 > uy0 and ux1 > ux0:
        img[uy0:uy1, ux0:ux1] = 255

    # Lower gate bar (dy = +180 HR px): width 200 HR px, right-shifted (-60 to +140)
    gy_dn = target_cy_hr + 180
    dy0, dy1 = max(0, gy_dn - local_gw // 2), min(h, gy_dn + local_gw // 2)
    dx0, dx1 = max(0, target_cx_hr - 60), min(w, target_cx_hr + 140)
    if dy1 > dy0 and dx1 > dx0:
        img[dy0:dy1, dx0:dx1] = 255

    # 5. Asymmetric contact pad: 30x30 HR px, offset to (+35, -20) HR px from target center
    contact_cx = target_cx_hr + 35
    contact_cy = target_cy_hr - 20
    contact_r = 15
    cx0, cx1 = max(0, contact_cx - contact_r), min(w, contact_cx + contact_r)
    cy0, cy1 = max(0, contact_cy - contact_r), min(h, contact_cy + contact_r)
    if cy1 > cy0 and cx1 > cx0:
        img[cy0:cy1, cx0:cx1] = 255

    return img


def draw_dram_scene(
    size: int = 10000,
    target_cx_hr: int = 5000,
    target_cy_hr: int = 5000,
    word_pitch_hr: float = 80.0,
    bit_pitch_hr: float = 80.0,
    line_width_hr: float = 20.0
) -> np.ndarray:
    """
    Generate DRAM memory array high-resolution continuous scene with V1.3 Asymmetric Landmark.
    Target-local landmark: 220x220 HR px framed tap pad with top-right offset cutout (+30, -30 HR px).
    """
    h = w = size
    img = np.zeros((h, w), dtype=np.uint8)

    # 1. Horizontal Word Lines (180 intensity)
    y = 0.0
    while y < h + word_pitch_hr:
        y0 = max(0, int(round(y - line_width_hr / 2.0)))
        y1 = min(h, int(round(y + line_width_hr / 2.0)))
        if y1 > y0:
            img[y0:y1, :] = 180
        y += word_pitch_hr

    # 2. Vertical Bit Lines (210 intensity)
    x = 0.0
    while x < w + bit_pitch_hr:
        x0 = max(0, int(round(x - line_width_hr / 2.0)))
        x1 = min(w, int(round(x + line_width_hr / 2.0)))
        if x1 > x0:
            img[:, x0:x1] = np.maximum(img[:, x0:x1], 210)
        x += bit_pitch_hr

    # 3. Contacts at intersections (255 intensity, 16 HR px diameter)
    contact_r = 8
    y = 0.0
    while y < h + word_pitch_hr:
        cy = int(round(y))
        x = 0.0
        while x < w + bit_pitch_hr:
            cx = int(round(x))
            cy0, cy1 = max(0, cy - contact_r), min(h, cy + contact_r)
            cx0, cx1 = max(0, cx - contact_r), min(w, cx + contact_r)
            if cy1 > cy0 and cx1 > cx0:
                img[cy0:cy1, cx0:cx1] = 255
            x += bit_pitch_hr
        y += word_pitch_hr

    # 4. Asymmetric Target-local landmark: 220x220 HR px framed tap pad with top-right offset cutout
    tap_half = 110
    ty0, ty1 = max(0, target_cy_hr - tap_half), min(h, target_cy_hr + tap_half)
    tx0, tx1 = max(0, target_cx_hr - tap_half), min(w, target_cx_hr + tap_half)
    if ty1 > ty0 and tx1 > tx0:
        img[ty0:ty1, tx0:tx1] = 255
        # Offset cutout centered at (+30, -30) HR px from target center
        cut_cx = target_cx_hr + 30
        cut_cy = target_cy_hr - 30
        cut_r = 45
        cy0, cy1 = max(0, cut_cy - cut_r), min(h, cut_cy + cut_r)
        cx0, cx1 = max(0, cut_cx - cut_r), min(w, cut_cx + cut_r)
        if cy1 > cy0 and cx1 > cx0:
            img[cy0:cy1, cx0:cx1] = 180

    return img


def generate_scene_and_pair(
    architecture: str = "FinFET",
    target_cx_search: float = 500.0,
    target_cy_search: float = 500.0,
    scale_ratio: float = 10.0,
    ref_size: int = 1000,
    search_size: int = 1000,
    variant_seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate physical scene, extract 1000x1000 reference crop, and downsample search image.
    """
    physical_scene_size = int(round(search_size * scale_ratio))

    cx_hr = int(round(target_cx_search * scale_ratio))
    cy_hr = int(round(target_cy_search * scale_ratio))

    if architecture.lower() == "dram":
        scene = draw_dram_scene(
            size=physical_scene_size,
            target_cx_hr=cx_hr,
            target_cy_hr=cy_hr
        )
    else:
        scene = draw_finfet_scene(
            size=physical_scene_size,
            target_cx_hr=cx_hr,
            target_cy_hr=cy_hr
        )

    # Crop reference (1000x1000) centered at target in high-res space
    ref_half = ref_size // 2
    left_hr = cx_hr - ref_half
    top_hr = cy_hr - ref_half
    right_hr = left_hr + ref_size
    bottom_hr = top_hr + ref_size

    # Crop with zero padding if near edges
    scene_h, scene_w = scene.shape
    crop_top = max(0, top_hr)
    crop_bottom = min(scene_h, bottom_hr)
    crop_left = max(0, left_hr)
    crop_right = min(scene_w, right_hr)

    ref_crop = np.zeros((ref_size, ref_size), dtype=np.uint8)
    dest_y0 = crop_top - top_hr
    dest_y1 = dest_y0 + (crop_bottom - crop_top)
    dest_x0 = crop_left - left_hr
    dest_x1 = dest_x0 + (crop_right - crop_left)

    if crop_bottom > crop_top and crop_right > crop_left:
        ref_crop[dest_y0:dest_y1, dest_x0:dest_x1] = scene[crop_top:crop_bottom, crop_left:crop_right]

    # Generate search image by downsampling full physical scene to search_size x search_size (13x faster than PIL)
    search_img = cv2.resize(scene, (search_size, search_size), interpolation=cv2.INTER_AREA)

    # Compute ground truth bbox and center in search image coordinates
    xmin = left_hr / scale_ratio
    ymin = top_hr / scale_ratio
    xmax = right_hr / scale_ratio
    ymax = bottom_hr / scale_ratio

    mapping_meta = {
        "target_center": {"x": float(target_cx_search), "y": float(target_cy_search)},
        "target_bbox": {
            "xmin": float(xmin),
            "ymin": float(ymin),
            "xmax": float(xmax),
            "ymax": float(ymax)
        },
        "high_res_bbox": {
            "xmin": int(left_hr),
            "ymin": int(top_hr),
            "xmax": int(right_hr),
            "ymax": int(bottom_hr)
        },
        "scale_ratio": float(scale_ratio)
    }

    return ref_crop, search_img, mapping_meta
