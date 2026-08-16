"""
Semiconductor Layer Decomposition and CAD Visualizer
===================================================
Extracts discrete structural/mask layers (Fins, Gates, Word Lines, Bit Lines, Contacts)
for FinFET and DRAM layouts, providing false-color representations and exploded 3D stacks.
"""

from typing import Dict, List, Tuple
import numpy as np
import cv2


_LAYER_COLORS = {
    "Substrate": (45, 45, 50),          # Charcoal
    "Silicon Fins": (240, 180, 50),     # Cyan / Azure
    "Gate Electrodes": (40, 100, 240),  # Bright Orange
    "Contact Pads": (70, 230, 90),      # Emerald Green
    "Word Lines": (60, 80, 240),        # Red / Orange
    "Bit Lines": (240, 150, 60),        # Sky Blue
    "Storage Contacts": (50, 220, 240), # Golden Yellow
    "Landmark Tap Pad": (200, 70, 240), # Neon Magenta
}


def false_color(mask: np.ndarray, color_bgr: Tuple[int, int, int]) -> np.ndarray:
    """Applies a specific BGR tint to a single grayscale mask (0-255)."""
    norm = mask.astype(np.float32) / 255.0
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = np.clip(norm * color_bgr[c], 0, 255).astype(np.uint8)
    return out


def decompose_finfet_layers(
    size: int = 1000,
    target_cx_hr: int = 500,
    target_cy_hr: int = 500,
    fin_pitch_hr: float = 90.0,
    fin_width_hr: float = 24.0,
    gate_width_hr: float = 30.0
) -> Dict[str, np.ndarray]:
    """Generates discrete layer masks for FinFET geometry."""
    h = w = size
    substrate = np.full((h, w), 35, dtype=np.uint8)
    fins = np.zeros((h, w), dtype=np.uint8)
    gates = np.zeros((h, w), dtype=np.uint8)
    contacts = np.zeros((h, w), dtype=np.uint8)

    # 1. Fins
    x = 0.0
    while x < w + fin_pitch_hr:
        x0 = max(0, int(round(x - fin_width_hr / 2.0)))
        x1 = min(w, int(round(x + fin_width_hr / 2.0)))
        if x1 > x0:
            fins[:, x0:x1] = 200
        x += fin_pitch_hr

    special_fin_x = target_cx_hr + int(round(fin_pitch_hr))
    sf_x0 = max(0, special_fin_x - 18)
    sf_x1 = min(w, special_fin_x + 18)
    sf_y0 = max(0, target_cy_hr - 60)
    sf_y1 = min(h, target_cy_hr + 60)
    if sf_x1 > sf_x0 and sf_y1 > sf_y0:
        fins[sf_y0:sf_y1, sf_x0:sf_x1] = 240

    # 2. Main Gate & Asymmetric Bars
    main_gate_y = target_cy_hr
    y0 = max(0, int(round(main_gate_y - gate_width_hr / 2.0)))
    y1 = min(h, int(round(main_gate_y + gate_width_hr / 2.0)))
    if y1 > y0:
        gates[y0:y1, :] = 220

    local_gw = int(round(gate_width_hr))
    gy_up = target_cy_hr - 180
    uy0, uy1 = max(0, gy_up - local_gw // 2), min(h, gy_up + local_gw // 2)
    ux0, ux1 = max(0, target_cx_hr - 160), min(w, target_cx_hr + 160)
    if uy1 > uy0 and ux1 > ux0:
        gates[uy0:uy1, ux0:ux1] = 255

    gy_dn = target_cy_hr + 180
    dy0, dy1 = max(0, gy_dn - local_gw // 2), min(h, gy_dn + local_gw // 2)
    dx0, dx1 = max(0, target_cx_hr - 60), min(w, target_cx_hr + 140)
    if dy1 > dy0 and dx1 > dx0:
        gates[dy0:dy1, dx0:dx1] = 255

    # 3. Contacts
    contact_cx = target_cx_hr + 35
    contact_cy = target_cy_hr - 20
    contact_r = 15
    cx0, cx1 = max(0, contact_cx - contact_r), min(w, contact_cx + contact_r)
    cy0, cy1 = max(0, contact_cy - contact_r), min(h, contact_cy + contact_r)
    if cy1 > cy0 and cx1 > cx0:
        contacts[cy0:cy1, cx0:cx1] = 255

    return {
        "Substrate": substrate,
        "Silicon Fins": fins,
        "Gate Electrodes": gates,
        "Contact Pads": contacts
    }


def decompose_dram_layers(
    size: int = 1000,
    target_cx_hr: int = 500,
    target_cy_hr: int = 500,
    word_pitch_hr: float = 80.0,
    bit_pitch_hr: float = 80.0,
    line_width_hr: float = 20.0
) -> Dict[str, np.ndarray]:
    """Generates discrete layer masks for DRAM array geometry."""
    h = w = size
    substrate = np.full((h, w), 35, dtype=np.uint8)
    word_lines = np.zeros((h, w), dtype=np.uint8)
    bit_lines = np.zeros((h, w), dtype=np.uint8)
    contacts = np.zeros((h, w), dtype=np.uint8)
    landmark = np.zeros((h, w), dtype=np.uint8)

    # 1. Word Lines
    y = 0.0
    while y < h + word_pitch_hr:
        y0 = max(0, int(round(y - line_width_hr / 2.0)))
        y1 = min(h, int(round(y + line_width_hr / 2.0)))
        if y1 > y0:
            word_lines[y0:y1, :] = 200
        y += word_pitch_hr

    # 2. Bit Lines
    x = 0.0
    while x < w + bit_pitch_hr:
        x0 = max(0, int(round(x - line_width_hr / 2.0)))
        x1 = min(w, int(round(x + line_width_hr / 2.0)))
        if x1 > x0:
            bit_lines[:, x0:x1] = 220
        x += bit_pitch_hr

    # 3. Contacts
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
                contacts[cy0:cy1, cx0:cx1] = 255
            x += bit_pitch_hr
        y += word_pitch_hr

    # 4. Landmark Tap Pad
    tap_half = 110
    ty0, ty1 = max(0, target_cy_hr - tap_half), min(h, target_cy_hr + tap_half)
    tx0, tx1 = max(0, target_cx_hr - tap_half), min(w, target_cx_hr + tap_half)
    if ty1 > ty0 and tx1 > tx0:
        landmark[ty0:ty1, tx0:tx1] = 255
        cut_cx = target_cx_hr + 30
        cut_cy = target_cy_hr - 30
        cut_r = 45
        cy0, cy1 = max(0, cut_cy - cut_r), min(h, cut_cy + cut_r)
        cx0, cx1 = max(0, cut_cx - cut_r), min(w, cut_cx + cut_r)
        if cy1 > cy0 and cx1 > cx0:
            landmark[cy0:cy1, cx0:cx1] = 0

    return {
        "Substrate": substrate,
        "Word Lines": word_lines,
        "Bit Lines": bit_lines,
        "Storage Contacts": contacts,
        "Landmark Tap Pad": landmark
    }


def build_exploded_stack(
    colored_layers: List[np.ndarray],
    layer_names: List[str],
    canvas_w: int = 1100,
    canvas_h: int = 680,
    layer_pitch_y: int = 95
) -> np.ndarray:
    """
    Renders an isometric 3D exploded stack perspective of semiconductor layers.
    """
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)
    n_layers = len(colored_layers)

    src_h, src_w = colored_layers[0].shape[:2]
    pts1 = np.float32([[0, 0], [src_w, 0], [0, src_h], [src_w, src_h]])

    target_w = 400
    target_h = 220
    pts2 = np.float32([
        [target_w * 0.35, 0],
        [target_w * 0.95, target_h * 0.22],
        [0, target_h * 0.78],
        [target_w * 0.60, target_h * 1.0]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    base_x = (canvas_w - int(target_w * 1.1)) // 2 + 80
    base_y = canvas_h - int(target_h * 1.3) - 20

    for i in range(n_layers):
        layer_img = colored_layers[i]
        warped = cv2.warpPerspective(
            layer_img, M, (int(target_w * 1.1), int(target_h * 1.1)),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        y_offset = base_y - i * layer_pitch_y
        x_offset = base_x

        wh, ww = warped.shape[:2]
        roi_y0 = max(0, y_offset)
        roi_y1 = min(canvas_h, y_offset + wh)
        roi_x0 = max(0, x_offset)
        roi_x1 = min(canvas_w, x_offset + ww)

        src_y0 = roi_y0 - y_offset
        src_y1 = src_y0 + (roi_y1 - roi_y0)
        src_x0 = roi_x0 - x_offset
        src_x1 = src_x0 + (roi_x1 - roi_x0)

        if roi_y1 > roi_y0 and roi_x1 > roi_x0:
            patch = warped[src_y0:src_y1, src_x0:src_x1]
            mask = (patch.sum(axis=2) > 20)[:, :, None]
            canvas_roi = canvas[roi_y0:roi_y1, roi_x0:roi_x1]
            canvas[roi_y0:roi_y1, roi_x0:roi_x1] = np.where(mask, patch, canvas_roi)

        # Draw layer label and indicator line
        label_text = f"Layer {i+1}: {layer_names[i]}"
        cv2.putText(
            canvas, label_text,
            (x_offset - 260, y_offset + target_h // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (225, 230, 240), 2, cv2.LINE_AA
        )
        cv2.line(
            canvas,
            (x_offset - 30, y_offset + target_h // 2 + 5),
            (x_offset + 30, y_offset + target_h // 2 + 5),
            (120, 130, 150), 1, cv2.LINE_AA
        )

    return canvas
