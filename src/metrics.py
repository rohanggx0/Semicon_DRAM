"""
Evaluation Metrics & Performance Benchmarking Module (V1.4 Structured Metrics)
"""

import math
from typing import List, Dict, Any
import numpy as np


def compute_euclidean_error(
    pred_x: float, pred_y: float,
    true_x: float, true_y: float
) -> float:
    """Calculate Euclidean distance error: sqrt((xpred - xtrue)^2 + (ypred - ytrue)^2)."""
    return float(math.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2))


def evaluate_batch_performance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute comprehensive structured metrics across a batch of evaluation cases:
    - Pass rates at 5px, 4px, 2px, 1px, and 0.5px (sub-pixel) thresholds
    - Mean, median, P95, std, min, and worst-case Euclidean error
    - Runtime statistics (mean, median, P95, min, max runtime in ms)
    - Architecture-wise breakdown (FinFET vs DRAM)
    """
    if not results:
        return {}

    errors = [r["euclidean_error"] for r in results]
    runtimes = [r["runtime_ms"] for r in results]

    num_cases = len(errors)

    pass_5px = sum(1 for e in errors if e <= 5.0) / num_cases * 100.0
    pass_4px = sum(1 for e in errors if e <= 4.0) / num_cases * 100.0
    pass_2px = sum(1 for e in errors if e <= 2.0) / num_cases * 100.0
    pass_1px = sum(1 for e in errors if e <= 1.0) / num_cases * 100.0
    pass_subpixel = sum(1 for e in errors if e <= 0.5) / num_cases * 100.0

    # Architecture breakdown
    finfet_res = [r for r in results if r.get("architecture", "").lower() == "finfet"]
    dram_res = [r for r in results if r.get("architecture", "").lower() == "dram"]

    finfet_errs = [r["euclidean_error"] for r in finfet_res] if finfet_res else []
    dram_errs = [r["euclidean_error"] for r in dram_res] if dram_res else []

    return {
        "total_test_cases": num_cases,
        "pass_rates_percent": {
            "threshold_5px": float(pass_5px),
            "threshold_4px": float(pass_4px),
            "threshold_2px": float(pass_2px),
            "threshold_1px": float(pass_1px),
            "threshold_subpixel_0_5px": float(pass_subpixel)
        },
        "euclidean_error_px": {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)),
            "std": float(np.std(errors)),
            "min": float(np.min(errors)),
            "worst_case_max": float(np.max(errors))
        },
        "architecture_breakdown": {
            "finfet": {
                "count": len(finfet_errs),
                "mean_error": float(np.mean(finfet_errs)) if finfet_errs else 0.0,
                "median_error": float(np.median(finfet_errs)) if finfet_errs else 0.0,
                "pass_5px": float(sum(1 for e in finfet_errs if e <= 5.0) / len(finfet_errs) * 100.0) if finfet_errs else 0.0,
                "pass_1px": float(sum(1 for e in finfet_errs if e <= 1.0) / len(finfet_errs) * 100.0) if finfet_errs else 0.0
            },
            "dram": {
                "count": len(dram_errs),
                "mean_error": float(np.mean(dram_errs)) if dram_errs else 0.0,
                "median_error": float(np.median(dram_errs)) if dram_errs else 0.0,
                "pass_5px": float(sum(1 for e in dram_errs if e <= 5.0) / len(dram_errs) * 100.0) if dram_errs else 0.0,
                "pass_1px": float(sum(1 for e in dram_errs if e <= 1.0) / len(dram_errs) * 100.0) if dram_errs else 0.0
            }
        },
        "runtime_ms": {
            "mean": float(np.mean(runtimes)),
            "median": float(np.median(runtimes)),
            "p95": float(np.percentile(runtimes, 95)),
            "min": float(np.min(runtimes)),
            "max": float(np.max(runtimes))
        }
    }
