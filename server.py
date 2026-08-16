"""
High-Performance Async Backend Server for Drift-Sense & FinFET Explorer
=======================================================================
Built with FastAPI to deliver sub-second data generation, non-blocking
asynchronous matcher execution, and responsive API endpoints.
"""

import base64
import io
import json
import math
import os
import sys
import time
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.geometry import generate_scene_and_pair
from src.sem_effects import apply_full_sem_pipeline
from src.matcher import localize_reference_in_search
from src.metrics import compute_euclidean_error
from src.layer_visualizer import (
    decompose_finfet_layers,
    decompose_dram_layers,
    false_color,
    build_exploded_stack,
    _LAYER_COLORS
)

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
    import torch
    _AI_MODEL = SEMRestorationUNet()
    _AI_MODEL.eval()
except Exception:
    HAS_AI = False
    _AI_MODEL = None


app = FastAPI(title="Drift-Sense SEM Explorer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

# Static files directory
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse("", status_code=204)


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "has_ai": HAS_AI,
        "has_pcm": HAS_PCM,
        "has_srm": HAS_SRM
    }


# ── In-Memory Caches ────────────────────────────────────────────────────────
_CACHE: Dict[str, Any] = {}
_SCENE_CACHE: Dict[str, Any] = {}


def arr_to_base64_fast(arr: np.ndarray) -> str:
    """Encodes a uint8 numpy array (grayscale or RGB) to Base64 JPEG in ~2ms."""
    _, enc = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("utf-8")


def arr_to_base64_png(arr: np.ndarray) -> str:
    """Encodes a uint8 numpy array (grayscale or RGB) to Base64 PNG for lossless downloads."""
    _, enc = cv2.imencode('.png', arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return "data:image/png;base64," + base64.b64encode(enc.tobytes()).decode("utf-8")


# ── Pydantic Request Models ──────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    architecture: str = "DRAM"
    target_x: float = 500.0
    target_y: float = 500.0
    scale_ratio: float = 10.0
    rotation_deg: float = 0.0
    blur_sigma: float = 0.6
    edge_bloom: float = 0.35
    shot_dose: float = 250.0
    gaussian_std: float = 5.0
    charging_strength: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    seed: int = 42


class MatchRequest(BaseModel):
    architecture: str = "DRAM"
    target_x: float = 500.0
    target_y: float = 500.0
    scale_ratio: float = 10.0
    rotation_deg: float = 0.0
    blur_sigma: float = 0.6
    edge_bloom: float = 0.35
    shot_dose: float = 250.0
    gaussian_std: float = 5.0
    charging_strength: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    seed: int = 42
    matcher: str = "Mode D"
    use_ai: bool = False


# ── Core Generation Helper with Scene Caching ────────────────────────────────
def _generate_pair_internal(req: GenerateRequest) -> Dict[str, Any]:
    scene_key = f"{req.architecture}_{req.target_x}_{req.target_y}_{req.scale_ratio}_{req.seed}"
    if scene_key in _SCENE_CACHE:
        raw_ref, raw_search, mapping_meta = _SCENE_CACHE[scene_key]
    else:
        raw_ref, raw_search, mapping_meta = generate_scene_and_pair(
            architecture=req.architecture,
            target_cx_search=req.target_x,
            target_cy_search=req.target_y,
            scale_ratio=req.scale_ratio,
            ref_size=1000,
            search_size=1000,
            variant_seed=req.seed
        )
        if len(_SCENE_CACHE) > 64:
            _SCENE_CACHE.clear()
        _SCENE_CACHE[scene_key] = (raw_ref, raw_search, mapping_meta)

    ref_sem = apply_full_sem_pipeline(
        raw_ref,
        edge_bloom=req.edge_bloom,
        blur_sigma=req.blur_sigma,
        shot_dose=req.shot_dose * 1.5,
        gaussian_std=req.gaussian_std * 0.5,
        charging_strength=0.0,
        contrast=req.contrast,
        gamma=req.gamma,
        rotation_deg=0.0,
        seed=req.seed
    )

    search_sem = apply_full_sem_pipeline(
        raw_search,
        edge_bloom=req.edge_bloom,
        blur_sigma=req.blur_sigma * 1.15,
        shot_dose=req.shot_dose,
        gaussian_std=req.gaussian_std,
        charging_strength=req.charging_strength,
        contrast=req.contrast,
        gamma=req.gamma,
        rotation_deg=req.rotation_deg,
        seed=req.seed + 1000
    )

    rad = math.radians(-req.rotation_deg)
    cx, cy = 499.5, 499.5
    dx = req.target_x - cx
    dy = req.target_y - cy
    gt_x_rot = cx + dx * math.cos(rad) - dy * math.sin(rad)
    gt_y_rot = cy + dx * math.sin(rad) + dy * math.cos(rad)

    return {
        "ref_sem": ref_sem,
        "search_sem": search_sem,
        "gt_x": float(np.clip(gt_x_rot, 0.0, 1000.0)),
        "gt_y": float(np.clip(gt_y_rot, 0.0, 1000.0)),
        "unrotated_tx": req.target_x,
        "unrotated_ty": req.target_y,
        "scale": req.scale_ratio,
        "rotation": req.rotation_deg,
        "arch": req.architecture,
        "seed": req.seed
    }


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the single-page dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard loading...</h1>")


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    """Generates clean & SEM reference/search images with ground truth metadata."""
    t0 = time.perf_counter()
    pair = _generate_pair_internal(req)
    gen_time_ms = (time.perf_counter() - t0) * 1000.0

    ref_b64 = arr_to_base64_fast(pair["ref_sem"])
    search_b64 = arr_to_base64_fast(pair["search_sem"])

    # Store in memory for instant matching
    cache_key = f"{req.architecture}_{req.target_x}_{req.target_y}_{req.scale_ratio}_{req.rotation_deg}_{req.seed}"
    _CACHE[cache_key] = pair

    return {
        "status": "success",
        "gen_time_ms": round(gen_time_ms, 2),
        "gt_x": round(pair["gt_x"], 3),
        "gt_y": round(pair["gt_y"], 3),
        "scale_ratio": req.scale_ratio,
        "rotation_deg": req.rotation_deg,
        "ref_image": ref_b64,
        "search_image": search_b64,
        "cache_key": cache_key
    }


@app.post("/api/match")
async def api_match(req: MatchRequest):
    """Executes a localization algorithm asynchronously and returns target coordinates."""
    t0 = time.perf_counter()

    cache_key = f"{req.architecture}_{req.target_x}_{req.target_y}_{req.scale_ratio}_{req.rotation_deg}_{req.seed}"
    if cache_key in _CACHE:
        pair = _CACHE[cache_key]
    else:
        gen_req = GenerateRequest(**req.dict(exclude={"matcher", "use_ai"}))
        pair = _generate_pair_internal(gen_req)
        _CACHE[cache_key] = pair

    ref_img = pair["ref_sem"]
    search_img = pair["search_sem"]

    # AI Pre-filtering if requested
    ai_applied = False
    if req.use_ai and HAS_AI and _AI_MODEL is not None:
        try:
            import torch
            with torch.no_grad():
                t_in = torch.from_numpy(search_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                t_out = _AI_MODEL(t_in).squeeze().cpu().numpy()
                search_img = np.clip(t_out * 255.0, 0, 255).astype(np.uint8)
                ai_applied = True
        except Exception:
            pass

    pred_x, pred_y, pred_scale, pred_rot, conf, score = 500.0, 500.0, 10.0, 0.0, 0.0, 0.0

    try:
        if req.matcher == "Mode D":
            loc_res = localize_reference_in_search(ref_img, search_img, scoring_mode="D", architecture=req.architecture)
            pred_x, pred_y = loc_res["predicted_center"]["x"], loc_res["predicted_center"]["y"]
            pred_scale = loc_res.get("best_scale", 10.0)
            pred_rot = loc_res.get("best_rotation_deg", 0.0)
            conf = loc_res.get("confidence", 0.0)
            score = loc_res.get("score_composite", 0.0)
        elif req.matcher == "Mode C":
            loc_res = localize_reference_in_search(ref_img, search_img, scoring_mode="C", architecture=req.architecture)
            pred_x, pred_y = loc_res["predicted_center"]["x"], loc_res["predicted_center"]["y"]
            pred_scale = loc_res.get("best_scale", 10.0)
            pred_rot = loc_res.get("best_rotation_deg", 0.0)
            conf = loc_res.get("confidence", 0.0)
            score = loc_res.get("score_composite", 0.0)
        elif req.matcher == "Mode A":
            loc_res = localize_reference_in_search(ref_img, search_img, scoring_mode="A", architecture=req.architecture)
            pred_x, pred_y = loc_res["predicted_center"]["x"], loc_res["predicted_center"]["y"]
            pred_scale = loc_res.get("best_scale", 10.0)
            pred_rot = loc_res.get("best_rotation_deg", 0.0)
            conf = loc_res.get("confidence", 0.0)
            score = loc_res.get("score_composite", 0.0)
        elif HAS_PCM and "Phase Correlation" in req.matcher:
            pcm = PhaseCorrelationMatcher()
            cands = pcm.search_candidates(ref_img, search_img, top_k=1)
            if cands:
                best = cands[0]
                pred_x = float(best.get("center_x", best.get("x", 500.0)))
                pred_y = float(best.get("center_y", best.get("y", 500.0)))
                pred_scale = float(best.get("scale_ratio", best.get("scale", 10.0)))
                pred_rot = float(best.get("rotation_deg", best.get("rotation", 0.0)))
                score = float(best.get("score_combined", best.get("score", 0.0)))
                conf = float(best.get("confidence", score))
        elif HAS_SRM and "ScaleRotationMatcher" in req.matcher:
            srm = ScaleRotationMatcher()
            cands = srm.search_candidates(ref_img, search_img, top_k=1)
            if cands:
                best = cands[0]
                pred_x = float(best.get("center_x", best.get("x", 500.0)))
                pred_y = float(best.get("center_y", best.get("y", 500.0)))
                pred_scale = float(best.get("scale_ratio", best.get("scale", 10.0)))
                pred_rot = float(best.get("rotation_deg", best.get("rotation", 0.0)))
                score = float(best.get("score_combined", best.get("score", 0.0)))
                conf = 1.0
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    err_dx = pred_x - pair["gt_x"]
    err_dy = pred_y - pair["gt_y"]
    euclid_err = compute_euclidean_error(pred_x, pred_y, pair["gt_x"], pair["gt_y"])

    return {
        "status": "success",
        "predicted_x": round(pred_x, 3),
        "predicted_y": round(pred_y, 3),
        "gt_x": round(pair["gt_x"], 3),
        "gt_y": round(pair["gt_y"], 3),
        "err_dx": round(err_dx, 3),
        "err_dy": round(err_dy, 3),
        "euclidean_error_px": round(euclid_err, 3),
        "is_success": euclid_err <= 5.0,
        "estimated_scale": round(pred_scale, 2),
        "estimated_rotation": round(pred_rot, 2),
        "confidence": round(conf, 3),
        "score": round(score, 3),
        "ai_applied": ai_applied,
        "latency_ms": round(latency_ms, 1)
    }


@app.get("/api/layers")
async def api_layers(architecture: str = "DRAM", patch_size: int = 1000):
    """Computes false-color masks and isometric 3D exploded stack."""
    if architecture == "FinFET":
        layers = decompose_finfet_layers(size=patch_size)
    else:
        layers = decompose_dram_layers(size=patch_size)

    layer_cards = []
    colored_layers = []
    layer_names = list(layers.keys())

    for name, mask in layers.items():
        color = _LAYER_COLORS.get(name, (200, 200, 200))
        colored = false_color(mask, color)
        colored_layers.append(colored)
        layer_cards.append({
            "name": name,
            "image": arr_to_base64_fast(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        })

    exploded_canvas = build_exploded_stack(colored_layers, layer_names)
    exploded_b64 = arr_to_base64_fast(cv2.cvtColor(exploded_canvas, cv2.COLOR_BGR2RGB))

    return {
        "status": "success",
        "architecture": architecture,
        "layers": layer_cards,
        "exploded_stack": exploded_b64
    }


@app.get("/api/family")
async def api_family(architecture: str = "DRAM", seed: int = 42):
    """Generates 1 Reference crop vs 5 SEM search acquisition variants."""
    variants_spec = [
        {"name": "1. Nominal Clean SEM", "dose": 500.0, "blur": 0.4, "g_std": 3.0, "charging": 0.0, "rot": 0.0},
        {"name": "2. Low Dose Shot Noise", "dose": 35.0, "blur": 0.5, "g_std": 12.0, "charging": 0.0, "rot": 0.0},
        {"name": "3. Beam Blur & Astigmatism", "dose": 300.0, "blur": 1.8, "g_std": 6.0, "charging": 0.0, "rot": 0.0},
        {"name": "4. Local Charging Bleed", "dose": 250.0, "blur": 0.6, "g_std": 5.0, "charging": 0.85, "rot": 0.0},
        {"name": "5. Severe Drift & Rotation", "dose": 200.0, "blur": 0.7, "g_std": 7.0, "charging": 0.2, "rot": 2.5},
    ]

    base_req = GenerateRequest(architecture=architecture, seed=seed)
    ref_pair = _generate_pair_internal(base_req)
    ref_b64 = arr_to_base64_fast(ref_pair["ref_sem"])

    results = []
    for v in variants_spec:
        v_req = GenerateRequest(
            architecture=architecture,
            rotation_deg=v["rot"],
            blur_sigma=v["blur"],
            shot_dose=v["dose"],
            gaussian_std=v["g_std"],
            charging_strength=v["charging"],
            seed=seed
        )
        p = _generate_pair_internal(v_req)
        loc = localize_reference_in_search(ref_pair["ref_sem"], p["search_sem"], scoring_mode="D")
        px, py = loc["predicted_center"]["x"], loc["predicted_center"]["y"]
        err = compute_euclidean_error(px, py, p["gt_x"], p["gt_y"])

        # Render search with crosshair
        disp = cv2.cvtColor(p["search_sem"], cv2.COLOR_GRAY2RGB)
        cv2.circle(disp, (int(round(p["gt_x"])), int(round(p["gt_y"]))), 5, (50, 205, 50), 1, cv2.LINE_AA)
        cv2.circle(disp, (int(round(px)), int(round(py))), 4, (255, 60, 60), 2, cv2.LINE_AA)

        results.append({
            "name": v["name"],
            "image": arr_to_base64_fast(disp),
            "error_px": round(err, 2),
            "passed": err <= 5.0,
            "gt": f"({p['gt_x']:.1f}, {p['gt_y']:.1f})",
            "pred": f"({px:.1f}, {py:.1f})"
        })

    return {
        "status": "success",
        "reference_image": ref_b64,
        "variants": results
    }


def _eval_single_benchmark_sample(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ref_p = BASE_DIR / item["ref_path"]
    search_p = BASE_DIR / item["search_path"]
    if not (ref_p.exists() and search_p.exists()):
        return None

    ref_img = cv2.imread(str(ref_p), cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(str(search_p), cv2.IMREAD_GRAYSCALE)
    if ref_img is None or search_img is None:
        return None

    t0 = time.perf_counter()
    loc = localize_reference_in_search(ref_img, search_img, scoring_mode="D")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    px, py = loc["predicted_center"]["x"], loc["predicted_center"]["y"]
    tx, ty = item["true_x"], item["true_y"]
    err = compute_euclidean_error(px, py, tx, ty)

    return {
        "sample_id": item["sample_id"],
        "scale": item["scale_ratio"],
        "rotation": item["rotation_deg"],
        "noise": item.get("noise_level", "medium"),
        "true_coord": f"({tx:.1f}, {ty:.1f})",
        "pred_coord": f"({px:.1f}, {py:.1f})",
        "error_px": round(err, 3),
        "passed": err <= 5.0,
        "latency_ms": round(latency_ms, 1)
    }


@app.get("/api/benchmark/stream")
async def api_benchmark_stream():
    """Streams the 30-pair benchmark results in real-time as each sample completes."""
    manifest_path = BASE_DIR / "data" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="data/manifest.json not found.")

    manifest_items = json.loads(manifest_path.read_text(encoding="utf-8"))

    async def event_generator():
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(executor, _eval_single_benchmark_sample, item)
            for item in manifest_items
        ]

        results_so_far = []
        for fut in asyncio.as_completed(futures):
            res = await fut
            if res is not None:
                results_so_far.append(res)
                errors = [r["error_px"] for r in results_so_far]
                summary = {
                    "total_processed": len(results_so_far),
                    "total_samples": len(manifest_items),
                    "pass_rate_5px": round(sum(1 for e in errors if e <= 5.0) / len(errors) * 100.0, 1),
                    "mean_error": round(float(np.mean(errors)), 3),
                    "median_error": round(float(np.median(errors)), 3),
                    "p95_error": round(float(np.percentile(errors, 95)), 3),
                }
                payload = {
                    "type": "row",
                    "row": res,
                    "summary": summary
                }
                yield f"data: {json.dumps(payload)}\n\n"

        if results_so_far:
            results_so_far.sort(key=lambda x: x["sample_id"])
            # Auto-generate evaluation PDF report from live 30-pair results
            try:
                from generate_pdf_report import generate_dram30_evaluation_pdf
                generate_dram30_evaluation_pdf(results=results_so_far, summary=summary)
                pdf_ready = True
            except Exception as e:
                print(f"[PDF Gen Error] {e}")
                pdf_ready = False

            yield f"data: {json.dumps({'type': 'complete', 'summary': summary, 'pdf_generated': pdf_ready})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/benchmark/run")
async def api_benchmark_run():
    """Runs the full 30-pair standardized benchmark with parallel thread execution."""
    manifest_path = BASE_DIR / "data" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="data/manifest.json not found.")

    manifest_items = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    loop = asyncio.get_running_loop()
    futures = [
        loop.run_in_executor(executor, _eval_single_benchmark_sample, item)
        for item in manifest_items
    ]
    raw_results = await asyncio.gather(*futures)
    results = [r for r in raw_results if r is not None]

    if not results:
        return {"status": "error", "message": "No valid benchmark samples found."}

    # Sort results by sample_id
    results.sort(key=lambda x: x["sample_id"])

    errors = [r["error_px"] for r in results]
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    p95_err = float(np.percentile(errors, 95))
    pass_rate = float(sum(1 for e in errors if e <= 5.0) / len(errors) * 100.0)

    summary = {
        "total_processed": len(results),
        "total_samples": len(results),
        "pass_rate_5px": round(pass_rate, 1),
        "mean_error": round(mean_err, 3),
        "median_error": round(median_err, 3),
        "p95_error": round(p95_err, 3),
    }

    # Auto-generate evaluation PDF report
    try:
        from generate_pdf_report import generate_dram30_evaluation_pdf
        generate_dram30_evaluation_pdf(results=results, summary=summary)
        pdf_ready = True
    except Exception as e:
        print(f"[PDF Gen Error] {e}")
        pdf_ready = False

    return {
        "status": "success",
        "total_samples": len(results),
        "pass_rate_5px": round(pass_rate, 1),
        "mean_error": round(mean_err, 3),
        "median_error": round(median_err, 3),
        "p95_error": round(p95_err, 3),
        "pdf_generated": pdf_ready,
        "rows": results
    }


@app.get("/api/reports/status")
async def get_report_status():
    """Returns the compilation status and metadata of the evaluation PDF report."""
    pdf_path = BASE_DIR / "DRAM_30_Evaluation_Report.pdf"
    summary_path = BASE_DIR / "results" / "metrics" / "dram30_latest_run_summary.json"
    
    if not pdf_path.exists():
        return {
            "exists": False,
            "filename": "DRAM_30_Evaluation_Report.pdf",
            "message": "Report not yet generated. Run the 30-Pair Benchmark to compile."
        }

    stat = pdf_path.stat()
    last_mod = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    size_kb = round(stat.st_size / 1024.0, 1)

    summary_info = {}
    if summary_path.exists():
        try:
            summary_info = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "exists": True,
        "filename": "DRAM_30_Evaluation_Report.pdf",
        "size_kb": size_kb,
        "last_modified": last_mod,
        "summary": summary_info
    }


@app.post("/api/reports/generate")
async def api_generate_report():
    """Explicitly triggers fresh compilation of the evaluation PDF report."""
    try:
        from generate_pdf_report import generate_dram30_evaluation_pdf
        pdf_path = generate_dram30_evaluation_pdf()
        stat = pdf_path.stat()
        return {
            "status": "success",
            "filename": "DRAM_30_Evaluation_Report.pdf",
            "size_kb": round(stat.st_size / 1024.0, 1),
            "last_modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")


@app.get("/api/reports/download")
async def download_pdf_report():
    """Downloads the compiled evaluation PDF report."""
    pdf_path = BASE_DIR / "DRAM_30_Evaluation_Report.pdf"
    if pdf_path.exists():
        return FileResponse(
            path=str(pdf_path),
            filename="DRAM_30_Evaluation_Report.pdf",
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=DRAM_30_Evaluation_Report.pdf"}
        )
    raise HTTPException(status_code=404, detail="Report PDF not found. Run generate_pdf_report.py first.")


@app.get("/api/reports/view")
async def view_pdf_report():
    """Inline view of the compiled evaluation PDF report for embedding in an iframe."""
    pdf_path = BASE_DIR / "DRAM_30_Evaluation_Report.pdf"
    if pdf_path.exists():
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            headers={"Content-Disposition": "inline; filename=DRAM_30_Evaluation_Report.pdf"}
        )
    raise HTTPException(status_code=404, detail="Report PDF not found.")


if __name__ == "__main__":
    import uvicorn
    print("Starting Drift-Sense High-Performance Dashboard on http://localhost:8000")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
