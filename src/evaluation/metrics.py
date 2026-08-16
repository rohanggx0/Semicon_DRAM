import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class Evaluator:
    """
    Evaluates localization performance metrics: Euclidean error, pass rates,
    runtime, group breakdowns (scale/rotation/noise), and plots visual overlays.
    """

    @staticmethod
    def compute_euclidean_error(x_pred: float, y_pred: float, x_true: float, y_true: float) -> float:
        """
        Computes Euclidean localization error in search image pixels.
        """
        return float(np.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2))

    @staticmethod
    def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates summary benchmark statistics from a list of per-sample results.
        """
        errors = np.array([r["error_px"] for r in results])
        runtimes = np.array([r["runtime_ms"] for r in results])

        n = len(errors)
        if n == 0:
            return {}

        summary = {
            "total_samples": n,
            "mean_error_px": float(np.mean(errors)),
            "median_error_px": float(np.median(errors)),
            "std_error_px": float(np.std(errors)),
            "min_error_px": float(np.min(errors)),
            "max_error_px": float(np.max(errors)),
            "pass_rate_le_5px": float(np.mean(errors <= 5.0) * 100.0),
            "pass_rate_le_4px": float(np.mean(errors <= 4.0) * 100.0),
            "pass_rate_le_2px": float(np.mean(errors <= 2.0) * 100.0),
            "pass_rate_le_1px": float(np.mean(errors <= 1.0) * 100.0),
            "pass_rate_subpixel_le_0_5px": float(np.mean(errors <= 0.5) * 100.0),
            "avg_runtime_ms": float(np.mean(runtimes))
        }

        return summary

    @staticmethod
    def plot_sample_result(
        ref_img: np.ndarray,
        search_img: np.ndarray,
        x_pred: float,
        y_pred: float,
        x_true: float,
        y_true: float,
        sample_id: int,
        save_path: str
    ):
        """
        Generates and saves visual overlay plot showing Reference target,
        Search image, Ground Truth crosshair (green), and Predicted crosshair (red).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Reference image
        axes[0].imshow(ref_img, cmap="gray")
        axes[0].set_title(f"Reference Target (100x Magnification)\nSample #{sample_id}")
        axes[0].axis("off")

        # Search image with overlays
        axes[1].imshow(search_img, cmap="gray")
        
        # Ground Truth crosshair (Green)
        axes[1].scatter([x_true], [y_true], c="lime", s=100, marker="+", linewidths=2, label="Ground Truth (x,y)")
        # Predicted crosshair (Red X)
        axes[1].scatter([x_pred], [y_pred], c="red", s=80, marker="x", linewidths=2, label="Predicted Target")
        
        # Circle search center tie-breaker reference point (500, 500)
        axes[1].scatter([500.0], [500.0], c="cyan", s=40, marker="o", label="Search Center (500,500)")

        error_px = Evaluator.compute_euclidean_error(x_pred, y_pred, x_true, y_true)
        axes[1].set_title(f"Search Image (10x Magnification)\nError = {error_px:.2f} px")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
