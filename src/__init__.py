"""
Drift-Sense Integrated Semiconductor Wafer Inspection & Navigation-Error Recovery Package.

Sub-modules:
  src.matching          – ScaleRotationMatcher (multi-scale ZNCC + Sobel), PhaseCorrelationMatcher
  src.localization      – AmbiguityResolver (center tie-breaker for periodic DRAM patterns)
  src.refinement        – SubpixelRefiner (quadratic 2D correlation surface fitting)
  src.degradation       – SEMDegradationEngine (Poisson shot noise, PSF blur, raster jitter)
  src.preprocessing     – SEMPreprocessor (DoG + Sobel gradient feature maps)
  src.evaluation        – Evaluator (euclidean error, pass@Npx, batch metrics)
"""

# ── Legacy top-level modules (direct flat files in src/) ──────────────────────
from src.matcher import (
    localize_reference_in_search,
    refine_subpixel_peak,
    extract_local_maxima,
)
from src.phase_correlation_matcher import PhaseCorrelationMatcher
from src.metrics import compute_euclidean_error, evaluate_batch_performance
from src.geometry import draw_finfet_scene, draw_dram_scene, generate_scene_and_pair
from src.sem_effects import apply_full_sem_pipeline

# ── Drift-Sense sub-module integrations ───────────────────────────────────────
from src.matching import ScaleRotationMatcher
from src.localization import AmbiguityResolver
from src.refinement import SubpixelRefiner
from src.degradation import SEMDegradationEngine
from src.preprocessing import SEMPreprocessor
from src.evaluation import Evaluator

__all__ = [
    # Legacy top-level
    "localize_reference_in_search",
    "refine_subpixel_peak",
    "extract_local_maxima",
    "PhaseCorrelationMatcher",
    "compute_euclidean_error",
    "evaluate_batch_performance",
    "draw_finfet_scene",
    "draw_dram_scene",
    "generate_scene_and_pair",
    "apply_full_sem_pipeline",
    # Drift-Sense sub-modules
    "ScaleRotationMatcher",
    "AmbiguityResolver",
    "SubpixelRefiner",
    "SEMDegradationEngine",
    "SEMPreprocessor",
    "Evaluator",
]
