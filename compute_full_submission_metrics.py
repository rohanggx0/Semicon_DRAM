import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(r"D:\SemiconFINFETwork")
metrics_json_path = BASE_DIR / "results" / "metrics" / "dram_ml_benchmark_summary.json"

with open(metrics_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ds1 = data["dataset_1_generated"]["per_pair_results"]
ds2 = data["dataset_2_engineered"]["per_pair_results"]

def compute_threshold_metrics(results, cv_key="cv_error_px", ml_key="ml_error_px", cv_time_key="cv_runtime_ms", ml_time_key="ml_runtime_ms"):
    cv_errs = np.array([r[cv_key] for r in results])
    ml_errs = np.array([r[ml_key] for r in results])
    
    cv_times = np.array([r[cv_time_key] for r in results])
    ml_times = np.array([r[ml_time_key] for r in results])

    def calc_stats(errs, times):
        N = len(errs)
        return {
            "total_pairs": N,
            "pass_5px": float(np.sum(errs <= 5.0) / N * 100.0),
            "pass_4px": float(np.sum(errs <= 4.0) / N * 100.0),
            "pass_2px": float(np.sum(errs <= 2.0) / N * 100.0),
            "pass_1px": float(np.sum(errs <= 1.0) / N * 100.0),
            "subpixel_0_5px": float(np.sum(errs <= 0.5) / N * 100.0),
            "mean_error_px": float(np.mean(errs)),
            "median_error_px": float(np.median(errs)),
            "worst_case_max_error_px": float(np.max(errs)),
            "avg_runtime_ms": float(np.mean(times))
        }

    return {
        "classical_cv": calc_stats(cv_errs, cv_times),
        "pytorch_ml": calc_stats(ml_errs, ml_times)
    }

metrics_ds1 = compute_threshold_metrics(ds1)
metrics_ds2 = compute_threshold_metrics(ds2)

print("=" * 90)
print("  MULTI-THRESHOLD SUBMISSION METRICS REPORT (5px, 4px, 2px, 1px, Sub-Pixel)")
print("=" * 90)

print("\n--- DATASET 1: Generated Synthetic DRAM (N=30) ---")
print("CLASSICAL CV:")
for k, v in metrics_ds1["classical_cv"].items():
    print(f"  {k:<25}: {v:.2f}" if isinstance(v, float) else f"  {k:<25}: {v}")

print("\nPYTORCH ML (U-Net):")
for k, v in metrics_ds1["pytorch_ml"].items():
    print(f"  {k:<25}: {v:.2f}" if isinstance(v, float) else f"  {k:<25}: {v}")

print("\n" + "-" * 90)
print("--- DATASET 2: Engineered DRAM_30 Benchmark (N=30) ---")
print("CLASSICAL CV:")
for k, v in metrics_ds2["classical_cv"].items():
    print(f"  {k:<25}: {v:.2f}" if isinstance(v, float) else f"  {k:<25}: {v}")

print("\nPYTORCH ML (U-Net):")
for k, v in metrics_ds2["pytorch_ml"].items():
    print(f"  {k:<25}: {v:.2f}" if isinstance(v, float) else f"  {k:<25}: {v}")

# Save detailed threshold summary json
out_json = BASE_DIR / "results" / "metrics" / "dram_ml_multi_threshold_summary.json"
out_json.write_text(json.dumps({
    "dataset_1_synthetic": metrics_ds1,
    "dataset_2_engineered": metrics_ds2
}, indent=2), encoding="utf-8")

print(f"\n[OK] Saved multi-threshold summary to: {out_json}")
