"""
Drift-Sense & FinFET Semiconductor Synthetic Data & Localization Explorer
========================================================================
A comprehensive interactive dashboard for synthetic SEM data generation,
multi-layer CAD decomposition, multi-algorithm matcher evaluation, and
robustness testing for the Applied Materials Drift-Sense problem statement.
"""

import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.geometry import generate_scene_and_pair
from src.sem_effects import apply_full_sem_pipeline
from src.matcher import localize_reference_in_search, refine_subpixel_peak
from src.metrics import compute_euclidean_error
from src.layer_visualizer import (
    decompose_finfet_layers,
    decompose_dram_layers,
    false_color,
    build_exploded_stack,
    _LAYER_COLORS
)

# Optional modular matchers
try:
    from src.phase_correlation_matcher import PhaseCorrelationMatcher
    HAS_PCM = True
except Exception:
    HAS_PCM = False

try:
    from src.matching.scale_rotation_matcher import ScaleRotationMatcher
    HAS_SRM = True
except Exception:
    HAS_SRM = False

try:
    from models.ai_restoration import SEMRestorationUNet
    HAS_AI = True
except Exception:
    HAS_AI = False


# ── Page Configuration & Theming ──────────────────────────────────────────────
st.set_page_config(
    page_title="Drift-Sense | Semiconductor SEM Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished semiconductor dashboard aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    .badge-pass {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────────────
if "seed" not in st.session_state:
    st.session_state.seed = 42

if "target_x" not in st.session_state:
    st.session_state.target_x = 520.0

if "target_y" not in st.session_state:
    st.session_state.target_y = 480.0


# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 SEM Controls")
    
    st.markdown("### 🏛️ Geometry & Physical Layout")
    architecture = st.selectbox("Architecture", ["FinFET", "DRAM"], index=0)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        target_x = st.number_input("Target X (Search px)", 200.0, 800.0, float(st.session_state.target_x), 10.0)
    with col_t2:
        target_y = st.number_input("Target Y (Search px)", 200.0, 800.0, float(st.session_state.target_y), 10.0)
    
    st.session_state.target_x = target_x
    st.session_state.target_y = target_y

    scale_ratio = st.slider(
        "Scale Ratio (Search:Ref FOV)", 8.5, 11.5, 10.0, 0.25,
        help="Physical FOV ratio. 10.0x = Ref is 1 um FOV (1 nm/px), Search is 10 um FOV (10 nm/px)."
    )
    rotation_deg = st.slider(
        "Relative Rotation (°)", -3.0, 3.0, 0.0, 0.5,
        help="Physical wafer rotation of search image relative to reference coordinate frame."
    )

    st.markdown("---")
    st.markdown("### ⚡ SEM Imaging Degradations")
    blur_sigma = st.slider("Beam Spot Blur (σ px)", 0.2, 2.5, 0.6, 0.1)
    edge_bloom = st.slider("Secondary Electron Edge Bloom", 0.0, 1.0, 0.35, 0.05)
    shot_dose = st.slider("Electron Shot Dose (Poisson)", 20.0, 1000.0, 250.0, 20.0, help="Lower dose = higher shot noise.")
    gaussian_std = st.slider("Detector Gaussian Noise (σ)", 0.0, 25.0, 5.0, 1.0)
    charging_strength = st.slider("Charging Drift / Bleed", 0.0, 1.0, 0.0, 0.05)
    contrast = st.slider("Contrast Multiplier", 0.5, 2.0, 1.0, 0.05)
    gamma = st.slider("Gamma Curve", 0.5, 2.0, 1.0, 0.05)

    st.markdown("---")
    st.markdown("### 🎲 Random Seed & Generation")
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        seed_input = st.number_input("Seed", 0, 999999, int(st.session_state.seed))
        st.session_state.seed = seed_input
    with col_s2:
        if st.button("🎲 Rand"):
            st.session_state.seed = int(np.random.randint(1, 999999))
            st.session_state.target_x = float(np.random.uniform(320.0, 680.0))
            st.session_state.target_y = float(np.random.uniform(320.0, 680.0))
            st.rerun()


# ── Core Synthetic Pair Generator ────────────────────────────────────────────
def generate_interactive_pair(
    arch: str, tx: float, ty: float, scale: float, rot: float,
    blur: float, bloom: float, dose: float, g_std: float,
    charging: float, cont: float, gam: float, s: int
) -> Dict[str, Any]:
    """Generates the clean scene, crops reference, downsamples search, and applies SEM degradations."""
    # 1. Clean continuous semiconductor scene
    raw_ref, raw_search, mapping_meta = generate_scene_and_pair(
        architecture=arch,
        target_cx_search=tx,
        target_cy_search=ty,
        scale_ratio=scale,
        ref_size=1000,
        search_size=1000,
        variant_seed=s
    )

    # 2. Reference SEM acquisition (fixed at 0 deg, clean dose)
    ref_sem = apply_full_sem_pipeline(
        raw_ref,
        edge_bloom=bloom,
        blur_sigma=blur,
        shot_dose=dose * 1.5,
        gaussian_std=g_std * 0.5,
        charging_strength=0.0,
        contrast=cont,
        gamma=gam,
        rotation_deg=0.0,
        seed=s
    )

    # 3. Search SEM acquisition (with rotation and search degradations)
    search_sem = apply_full_sem_pipeline(
        raw_search,
        edge_bloom=bloom,
        blur_sigma=blur * 1.15,
        shot_dose=dose,
        gaussian_std=g_std,
        charging_strength=charging,
        contrast=cont,
        gamma=gam,
        rotation_deg=rot,
        seed=s + 1000
    )

    # 4. Post-rotation ground truth coordinate transformation
    rad = math.radians(-rot)
    cx, cy = 499.5, 499.5
    dx = tx - cx
    dy = ty - cy
    gt_x_rot = cx + dx * math.cos(rad) - dy * math.sin(rad)
    gt_y_rot = cy + dx * math.sin(rad) + dy * math.cos(rad)

    return {
        "raw_ref": raw_ref,
        "raw_search": raw_search,
        "ref_sem": ref_sem,
        "search_sem": search_sem,
        "gt_x": float(np.clip(gt_x_rot, 0.0, 1000.0)),
        "gt_y": float(np.clip(gt_y_rot, 0.0, 1000.0)),
        "unrotated_tx": tx,
        "unrotated_ty": ty,
        "scale": scale,
        "rotation": rot,
        "arch": arch,
        "seed": s
    }


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ── Main Header ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 Drift-Sense & FinFET: Semiconductor SEM Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Multi-scale wafer navigation error recovery, physical CAD layer decomposition, and localization benchmark suite.</div>',
    unsafe_allow_html=True
)

tab_arena, tab_layers, tab_family, tab_bench, tab_export = st.tabs([
    "🎯 Live Matcher Arena",
    "🧱 CAD Layer Inspector",
    "🧪 Robustness Variants (Family)",
    "📊 Dataset Benchmark (30 Pairs)",
    "📑 Report & Deliverables"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Live Matcher Arena
# ══════════════════════════════════════════════════════════════════════════════
with tab_arena:
    pair_data = generate_interactive_pair(
        architecture, target_x, target_y, scale_ratio, rotation_deg,
        blur_sigma, edge_bloom, shot_dose, gaussian_std,
        charging_strength, contrast, gamma, st.session_state.seed
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.5, 1])
    with col_ctrl1:
        matcher_algo = st.selectbox(
            "Localization Engine",
            [
                "V1.7 Rotation-Geometric Matcher (Mode D - Full Multi-Modal)",
                "V1.7 Rotation-Geometric Matcher (Mode C - NCC + Landmark)",
                "V1.7 Rotation-Geometric Matcher (Mode A - Pure NCC)",
                "Phase Correlation Matcher (FFT Frequency Domain)" if HAS_PCM else None,
                "Multi-Scale ScaleRotationMatcher" if HAS_SRM else None
            ],
            index=0
        )
    with col_ctrl2:
        use_ai_filter = st.checkbox("Enable AI Restoration U-Net Pre-filter", value=False, disabled=not HAS_AI)
    with col_ctrl3:
        show_gt_box = st.checkbox("Show Ground Truth Overlay", value=True)

    # Optional AI Restoration Pre-filtering
    processed_search = pair_data["search_sem"]
    if use_ai_filter and HAS_AI:
        try:
            import torch
            with torch.no_grad():
                ai_model = SEMRestorationUNet()
                ai_model.eval()
                t_in = torch.from_numpy(pair_data["search_sem"].astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                t_out = ai_model(t_in).squeeze().cpu().numpy()
                processed_search = np.clip(t_out * 255.0, 0, 255).astype(np.uint8)
        except Exception as e:
            st.sidebar.warning(f"AI Pre-filter skipped: {e}")

    # Run selected localization engine
    t_start = time.perf_counter()
    pred_x, pred_y, pred_scale, pred_rot, conf, score = 500.0, 500.0, 10.0, 0.0, 0.0, 0.0

    try:
        if "Mode D" in matcher_algo:
            loc_res = localize_reference_in_search(pair_data["ref_sem"], processed_search, scoring_mode="D")
            pred_x, pred_y = loc_res["predicted_center"]["x"], loc_res["predicted_center"]["y"]
            pred_scale = loc_res.get("best_candidate", {}).get("scale", 10.0)
            pred_rot = loc_res.get("best_candidate", {}).get("rotation", 0.0)
            conf = loc_res.get("confidence", 0.0)
            score = loc_res.get("best_candidate", {}).get("score", 0.0)
        elif "Mode C" in matcher_algo:
            loc_res = localize_reference_in_search(pair_data["ref_sem"], processed_search, scoring_mode="C")
            pred_x, pred_y = loc_res["predicted_center"]["x"], loc_res["predicted_center"]["y"]
            pred_scale = loc_res.get("best_candidate", {}).get("scale", 10.0)
            pred_rot = loc_res.get("best_candidate", {}).get("rotation", 0.0)
            conf = loc_res.get("confidence", 0.0)
            score = loc_res.get("best_candidate", {}).get("score", 0.0)
        elif "Mode A" in matcher_algo:
            loc_res = localize_reference_in_search(pair_data["ref_sem"], processed_search, scoring_mode="A")
            pred_x, pred_y = loc_res["predicted_center"]["x"], loc_res["predicted_center"]["y"]
            pred_scale = loc_res.get("best_candidate", {}).get("scale", 10.0)
            pred_rot = loc_res.get("best_candidate", {}).get("rotation", 0.0)
            conf = loc_res.get("confidence", 0.0)
            score = loc_res.get("best_candidate", {}).get("score", 0.0)
        elif HAS_PCM and "Phase Correlation" in matcher_algo:
            pcm = PhaseCorrelationMatcher()
            cands = pcm.search_candidates(pair_data["ref_sem"], processed_search, top_k=1)
            if cands:
                best = cands[0]
                pred_x = float(best.get("center_x", best.get("x", 500.0)))
                pred_y = float(best.get("center_y", best.get("y", 500.0)))
                pred_scale = float(best.get("scale_ratio", best.get("scale", 10.0)))
                pred_rot = float(best.get("rotation_deg", best.get("rotation", 0.0)))
                score = float(best.get("score_combined", best.get("score", 0.0)))
                conf = float(best.get("confidence", score))
        elif HAS_SRM and "ScaleRotationMatcher" in matcher_algo:
            srm = ScaleRotationMatcher()
            cands = srm.search_candidates(pair_data["ref_sem"], processed_search, top_k=1)
            if cands:
                best = cands[0]
                pred_x = float(best.get("center_x", best.get("x", 500.0)))
                pred_y = float(best.get("center_y", best.get("y", 500.0)))
                pred_scale = float(best.get("scale_ratio", best.get("scale", 10.0)))
                pred_rot = float(best.get("rotation_deg", best.get("rotation", 0.0)))
                score = float(best.get("score_combined", best.get("score", 0.0)))
                conf = 1.0
    except Exception as e:
        st.error(f"Matcher execution error ({matcher_algo}): {e}")

    latency_ms = (time.perf_counter() - t_start) * 1000.0
    err_dx = pred_x - pair_data["gt_x"]
    err_dy = pred_y - pair_data["gt_y"]
    euclid_err = compute_euclidean_error(pred_x, pred_y, pair_data["gt_x"], pair_data["gt_y"])
    is_success = euclid_err <= 5.0

    # Render Visual Annotations on Search Image
    search_rgb = cv2.cvtColor(processed_search, cv2.COLOR_GRAY2RGB)

    if show_gt_box:
        # Green circle for Ground Truth center + 5 px tolerance ring
        cv2.circle(search_rgb, (int(round(pair_data["gt_x"])), int(round(pair_data["gt_y"]))), 5, (50, 205, 50), 1, cv2.LINE_AA)
        cv2.drawMarker(search_rgb, (int(round(pair_data["gt_x"])), int(round(pair_data["gt_y"]))), (50, 205, 50), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)


    # Red marker for Predicted Center
    cv2.circle(search_rgb, (int(round(pred_x)), int(round(pred_y))), 4, (255, 60, 60), 2, cv2.LINE_AA)
    cv2.drawMarker(search_rgb, (int(round(pred_x)), int(round(pred_y))), (255, 60, 60), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    # Residual displacement line
    if euclid_err > 0.5:
        cv2.line(
            search_rgb,
            (int(round(pair_data["gt_x"])), int(round(pair_data["gt_y"]))),
            (int(round(pred_x)), int(round(pred_y))),
            (255, 220, 0), 2, cv2.LINE_AA
        )

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Reference Image (100x Magnification — 1 nm/px, 1 µm FOV)")
        st.image(pair_data["ref_sem"], use_container_width=True, caption=f"Reference Crop (1000x1000 px) | Architecture: {architecture}")
    with col_v2:
        st.subheader(f"Search Image (10x Wide FOV — 10 nm/px, 10 µm FOV)")
        st.image(search_rgb, use_container_width=True, caption=f"Search Field (1000x1000 px) | True: ({pair_data['gt_x']:.1f}, {pair_data['gt_y']:.1f}) | Pred: ({pred_x:.1f}, {pred_y:.1f})")

    # Metrics Display Bar
    st.markdown("### 📈 Localization Metrics")
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        status_html = '<span class="badge-pass">PASS (≤5.0 px)</span>' if is_success else '<span class="badge-fail">FAIL (>5.0 px)</span>'
        st.markdown(f"**Status:** {status_html}", unsafe_allow_html=True)
        st.metric("Euclidean Error", f"{euclid_err:.3f} px")
    with mcol2:
        st.metric("Sub-pixel Shift (Δx, Δy)", f"Δx={err_dx:+.2f}, Δy={err_dy:+.2f} px")
    with mcol3:
        st.metric("Estimated Scale / Angle", f"{pred_scale:.2f}x | {pred_rot:+.1f}°")
    with mcol4:
        st.metric("Score & Confidence", f"Score: {score:.3f} | Conf: {conf:.2f}")
    with mcol5:
        st.metric("Execution Latency", f"{latency_ms:.1f} ms")

    # Download Bar
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.download_button("💾 Download reference.png", to_png_bytes(pair_data["ref_sem"]), "reference.png", "image/png")
    with dcol2:
        st.download_button("💾 Download search.png", to_png_bytes(pair_data["search_sem"]), "search.png", "image/png")
    with dcol3:
        meta_payload = {
            "architecture": architecture,
            "target_center_ground_truth": {"x": pair_data["gt_x"], "y": pair_data["gt_y"]},
            "predicted_center": {"x": pred_x, "y": pred_y},
            "scale_ratio": scale_ratio,
            "rotation_deg": rotation_deg,
            "euclidean_error_px": euclid_err,
            "sem_parameters": {
                "blur_sigma": blur_sigma,
                "shot_dose": shot_dose,
                "gaussian_std": gaussian_std,
                "charging_strength": charging_strength,
                "seed": st.session_state.seed
            }
        }
        st.download_button("💾 Download metadata.json", json.dumps(meta_payload, indent=2), "metadata.json", "application/json")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CAD Layer Inspector
# ══════════════════════════════════════════════════════════════════════════════
with tab_layers:
    st.markdown(
        "Real SEM inspection only captures the top exposed material surface with secondary electron emission. "
        "Internally, the semiconductor structure is composed of sequential lithography and etch mask layers. "
        "This inspector exposes each CAD layer individually in false-color, along with an exploded isometric 3D perspective."
    )

    layer_arch = st.radio("Inspect Architecture", ["FinFET", "DRAM"], horizontal=True)
    patch_size = st.slider("CAD Scene Extent (px)", 600, 1400, 1000, 100)

    if layer_arch == "FinFET":
        layers = decompose_finfet_layers(size=patch_size)
    else:
        layers = decompose_dram_layers(size=patch_size)

    layer_names = list(layers.keys())
    cols = st.columns(len(layer_names))

    colored_layers = []
    for idx, (name, mask) in enumerate(layers.items()):
        color = _LAYER_COLORS.get(name, (200, 200, 200))
        colored = false_color(mask, color)
        colored_layers.append(colored)
        with cols[idx]:
            st.image(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), caption=f"{name}", use_container_width=True)

    st.markdown("---")
    st.subheader("🧊 3D Isometric Exploded Layer Stack")
    exploded_canvas = build_exploded_stack(colored_layers, layer_names)
    st.image(cv2.cvtColor(exploded_canvas, cv2.COLOR_BGR2RGB), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Sample Family Robustness (5 Search Variants)
# ══════════════════════════════════════════════════════════════════════════════
with tab_family:
    st.markdown(
        "### 🧪 1 Reference vs 5 Physical Acquisition Variants\n"
        "Tests algorithm robustness when the same physical wafer region is imaged under 5 drastically different SEM conditions. "
        "A robust matcher should locate the reference consistently across all 5 variants."
    )

    variants_config = [
        {"name": "1. Nominal (Clean SEM)", "dose": 500.0, "blur": 0.4, "g_std": 3.0, "charging": 0.0, "rot": 0.0},
        {"name": "2. Low Dose (High Shot Noise)", "dose": 35.0, "blur": 0.5, "g_std": 12.0, "charging": 0.0, "rot": 0.0},
        {"name": "3. High Beam Blur / Astigmatism", "dose": 300.0, "blur": 1.8, "g_std": 6.0, "charging": 0.0, "rot": 0.0},
        {"name": "4. Local Charging Bleed / Streaks", "dose": 250.0, "blur": 0.6, "g_std": 5.0, "charging": 0.85, "rot": 0.0},
        {"name": "5. Severe Drift & Relative Rotation", "dose": 200.0, "blur": 0.7, "g_std": 7.0, "charging": 0.2, "rot": 2.5},
    ]

    selected_var_name = st.selectbox("Select Search Variant", [v["name"] for v in variants_config])
    cur_v = next(v for v in variants_config if v["name"] == selected_var_name)

    v_pair = generate_interactive_pair(
        architecture, target_x, target_y, 10.0, cur_v["rot"],
        cur_v["blur"], 0.35, cur_v["dose"], cur_v["g_std"],
        cur_v["charging"], 1.0, 1.0, st.session_state.seed
    )

    # Localize on this variant
    v_loc = localize_reference_in_search(v_pair["ref_sem"], v_pair["search_sem"], scoring_mode="D")
    vx, vy = v_loc["predicted_center"]["x"], v_loc["predicted_center"]["y"]
    v_err = compute_euclidean_error(vx, vy, v_pair["gt_x"], v_pair["gt_y"])

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.subheader("Reference (Fixed Clean 100x)")
        st.image(v_pair["ref_sem"], use_container_width=True)
    with vcol2:
        st.subheader(f"Search Variant: {selected_var_name}")
        v_disp = cv2.cvtColor(v_pair["search_sem"], cv2.COLOR_GRAY2RGB)
        cv2.circle(v_disp, (int(round(v_pair["gt_x"])), int(round(v_pair["gt_y"]))), 5, (50, 205, 50), 1, cv2.LINE_AA)
        cv2.circle(v_disp, (int(round(vx)), int(round(vy))), 4, (255, 60, 60), 2, cv2.LINE_AA)
        cv2.drawMarker(v_disp, (int(round(vx)), int(round(vy))), (255, 60, 60), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        st.image(v_disp, use_container_width=True)

    if v_err <= 5.0:
        st.success(f"✅ Matcher Successful on `{selected_var_name}` — Error: {v_err:.2f} px (within 5.0 px tolerance).")
    else:
        st.warning(f"⚠️ Matcher Exceeded Tolerance on `{selected_var_name}` — Error: {v_err:.2f} px (> 5.0 px).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Dataset Benchmark (30 Pairs)
# ══════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown("### 📊 Standardized 30-Pair DRAM/FinFET Benchmark")
    manifest_path = BASE_DIR / "data" / "manifest.json"

    if manifest_path.exists():
        try:
            manifest_items = json.loads(manifest_path.read_text(encoding="utf-8"))
            st.info(f"Loaded `{len(manifest_items)}` evaluation pairs from `data/manifest.json`.")

            if st.button("🚀 Run Full 30-Pair Benchmark Live"):
                with st.spinner("Evaluating 30 image pairs with Mode D Localization Engine..."):
                    results = []
                    prog_bar = st.progress(0)

                    for idx, item in enumerate(manifest_items):
                        ref_p = BASE_DIR / item["ref_path"]
                        search_p = BASE_DIR / item["search_path"]

                        if ref_p.exists() and search_p.exists():
                            ref_img = np.asarray(Image.open(ref_p).convert("L"), dtype=np.uint8)
                            search_img = np.asarray(Image.open(search_p).convert("L"), dtype=np.uint8)
                            loc = localize_reference_in_search(ref_img, search_img, scoring_mode="D")
                            px, py = loc["predicted_center"]["x"], loc["predicted_center"]["y"]
                            tx, ty = item["true_x"], item["true_y"]
                            err = compute_euclidean_error(px, py, tx, ty)

                            results.append({
                                "Sample ID": item["sample_id"],
                                "Scale": item["scale_ratio"],
                                "Rotation": item["rotation_deg"],
                                "Noise": item.get("noise_level", "medium"),
                                "True (x, y)": f"({tx:.1f}, {ty:.1f})",
                                "Pred (x, y)": f"({px:.1f}, {py:.1f})",
                                "Error (px)": round(err, 3),
                                "Passed (≤5 px)": "✅ Yes" if err <= 5.0 else "❌ No"
                            })
                        prog_bar.progress((idx + 1) / len(manifest_items))

                    errors = [r["Error (px)"] for r in results]
                    mean_err = np.mean(errors)
                    median_err = np.median(errors)
                    p95_err = np.percentile(errors, 95)
                    hit_rate = sum(1 for e in errors if e <= 5.0) / len(errors) * 100.0

                    st.markdown("#### 🎯 Overall Benchmark Summary")
                    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                    with bcol1:
                        st.metric("Success Rate (≤ 5.0 px)", f"{hit_rate:.1f}%")
                    with bcol2:
                        st.metric("Mean Euclidean Error", f"{mean_err:.3f} px")
                    with bcol3:
                        st.metric("Median Error", f"{median_err:.3f} px")
                    with bcol4:
                        st.metric("95th Percentile Error", f"{p95_err:.3f} px")

                    st.dataframe(results, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading manifest: {e}")
    else:
        st.warning("`data/manifest.json` not found. Run `python regenerate_synthetic_dataset.py` to generate the 30-pair dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Report & Deliverables
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("### 📑 PDF Technical Evaluation Reports & Submission Package")
    st.markdown(
        "Generate publication-ready PDF evaluation reports with complete mathematical formulation, "
        "confusion matrices, sub-pixel error histograms, and packaging tools."
    )

    pdf_path = BASE_DIR / "DRAM_30_Evaluation_Report.pdf"
    if pdf_path.exists():
        st.success("✅ `DRAM_30_Evaluation_Report.pdf` is generated and available for download.")
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download DRAM_30_Evaluation_Report.pdf", f.read(), "DRAM_30_Evaluation_Report.pdf", "application/pdf")
    else:
        st.info("Run `python generate_pdf_report.py` to compile the full technical PDF report.")

    st.markdown("---")
    st.markdown("### 📦 CLI Quick Commands")
    st.code("""
# 1. Regenerate 30 DRAM pairs with exact manifest coordinates
python regenerate_synthetic_dataset.py

# 2. Run batch benchmark evaluation with live metrics
python evaluate_dram30_live.py

# 3. Compile full publication-grade PDF report
python generate_pdf_report.py

# 4. Generate final submission zip archive
python create_submission_package.py
    """, language="bash")
