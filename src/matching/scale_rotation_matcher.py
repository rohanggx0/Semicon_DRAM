import numpy as np
# pyrefly: ignore [missing-import]
import cv2
from typing import List, Dict, Any, Tuple

from src.preprocessing.sem_filters import SEMPreprocessor

class ScaleRotationMatcher:
    """
    Multi-Scale, Multi-Rotation Zero-Mean Normalized Cross-Correlation (ZNCC)
    and Sobel Edge Feature Matcher for SEM Wafer Localization.

    V1.1 changes:
    - Coarse search now includes macro envelope channel (30% weight) to break
      DRAM periodic cell aliasing at the scale-selection stage.
    - Top coarse candidates raised from 8 → 14 to avoid missing correct scale.
    - Fine search local peak threshold lowered from 90% → 85% to surface more
      periodic alias peaks so AmbiguityResolver can pick closest-to-center.
    """

    def __init__(
        self,
        scale_min: float = 8.5,
        scale_max: float = 11.5,
        scale_step: float = 0.25,
        rotation_min: float = -2.5,
        rotation_max: float = 2.5,
        rotation_step: float = 0.5
    ):
        self.scale_range = np.arange(scale_min, scale_max + 1e-5, scale_step)
        self.rotation_range = np.arange(rotation_min, rotation_max + 1e-5, rotation_step)
        self.preprocessor = SEMPreprocessor()

    def get_scaled_rotated_template(
        self,
        reference_img: np.ndarray,
        scale_ratio: float,
        angle_deg: float
    ) -> np.ndarray:
        """
        Resizes 1000x1000 Reference template to match target scale ratio in Search domain,
        and applies requested rotation angle.
        """
        target_size = round(1000.0 / scale_ratio)
        target_size = max(10, min(900, target_size))

        scaled_tpl = cv2.resize(reference_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        if abs(angle_deg) > 1e-4:
            h, w = scaled_tpl.shape
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            scaled_tpl = cv2.warpAffine(scaled_tpl, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        return scaled_tpl

    def search_candidates(
        self,
        reference_img: np.ndarray,
        search_img: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes fast Coarse-to-Fine multi-pyramid template search across scale and rotation space.
        Coarse stage evaluates 2x downsampled image for fast global candidate selection.
        Fine stage refines correlation peak on full 1000x1000 resolution.
        """
        # Step 1: Preprocess full resolution maps
        ref_dog = self.preprocessor.preprocess(reference_img, method="dog")
        search_dog = self.preprocessor.preprocess(search_img, method="dog")

        ref_edge = self.preprocessor.preprocess(reference_img, method="sobel")
        search_edge = self.preprocessor.preprocess(search_img, method="sobel")

        ref_macro = self.preprocessor.preprocess(reference_img, method="macro")
        search_macro = self.preprocessor.preprocess(search_img, method="macro")

        # Step 2: Prepare 2x Downsampled Coarse Images
        sh, sw = search_dog.shape
        coarse_search_dog   = cv2.resize(search_dog,   (sw // 2, sh // 2), interpolation=cv2.INTER_AREA)
        coarse_search_edge  = cv2.resize(search_edge,  (sw // 2, sh // 2), interpolation=cv2.INTER_AREA)
        # Macro envelope added at coarse stage — critical for breaking DRAM periodic cell aliasing
        coarse_search_macro = cv2.resize(search_macro, (sw // 2, sh // 2), interpolation=cv2.INTER_AREA)

        coarse_ref_dog   = cv2.resize(ref_dog,   (500, 500), interpolation=cv2.INTER_AREA)
        coarse_ref_edge  = cv2.resize(ref_edge,  (500, 500), interpolation=cv2.INTER_AREA)
        coarse_ref_macro = cv2.resize(ref_macro, (500, 500), interpolation=cv2.INTER_AREA)

        coarse_candidates = []

        # Global Coarse Search Loop over Scale & Rotation
        for scale in self.scale_range:
            for angle in self.rotation_range:
                # 2x coarse template size
                target_size = int(round(500.0 / scale))
                target_size = max(10, min(450, target_size))

                tpl_d = cv2.resize(coarse_ref_dog,   (target_size, target_size), interpolation=cv2.INTER_AREA)
                tpl_e = cv2.resize(coarse_ref_edge,  (target_size, target_size), interpolation=cv2.INTER_AREA)
                tpl_m = cv2.resize(coarse_ref_macro, (target_size, target_size), interpolation=cv2.INTER_AREA)

                if abs(angle) > 1e-4:
                    h_c, w_c = tpl_d.shape
                    M = cv2.getRotationMatrix2D((w_c / 2.0, h_c / 2.0), angle, 1.0)
                    tpl_d = cv2.warpAffine(tpl_d, M, (w_c, h_c), borderMode=cv2.BORDER_REFLECT)
                    tpl_e = cv2.warpAffine(tpl_e, M, (w_c, h_c), borderMode=cv2.BORDER_REFLECT)
                    tpl_m = cv2.warpAffine(tpl_m, M, (w_c, h_c), borderMode=cv2.BORDER_REFLECT)

                th_c, tw_c = tpl_d.shape
                sh_c, sw_c = coarse_search_dog.shape

                if th_c >= sh_c or tw_c >= sw_c:
                    continue

                res_d = cv2.matchTemplate(coarse_search_dog,   tpl_d, cv2.TM_CCOEFF_NORMED)
                res_e = cv2.matchTemplate(coarse_search_edge,  tpl_e, cv2.TM_CCOEFF_NORMED)
                res_m = cv2.matchTemplate(coarse_search_macro, tpl_m, cv2.TM_CCOEFF_NORMED)
                # Macro envelope breaks DRAM periodic aliasing; weighted at 30% at coarse stage
                res_c = 0.45 * res_d + 0.25 * res_e + 0.30 * res_m

                _, max_val, _, _ = cv2.minMaxLoc(res_c)
                coarse_candidates.append({
                    "scale": scale,
                    "angle": angle,
                    "score": max_val
                })

        # Rank coarse candidates — use top-14 to ensure correct scale neighbourhood is always included
        coarse_candidates.sort(key=lambda c: c["score"], reverse=True)
        top_param_pairs = coarse_candidates[:14]

        # Step 3: Fine Search on Full Resolution for Top Parameter Neighborhoods
        fine_candidates = []

        for param in top_param_pairs:
            best_s = param["scale"]
            best_a = param["angle"]

            # Local fine sampling around best coarse parameters
            fine_scales = [best_s - 0.25, best_s, best_s + 0.25]
            fine_angles = [best_a - 0.5, best_a, best_a + 0.5]

            for scale in fine_scales:
                if scale < self.scale_range[0] or scale > self.scale_range[-1]:
                    continue
                for angle in fine_angles:
                    if angle < self.rotation_range[0] or angle > self.rotation_range[-1]:
                        continue

                    tpl_dog   = self.get_scaled_rotated_template(ref_dog,   scale, angle)
                    tpl_edge  = self.get_scaled_rotated_template(ref_edge,  scale, angle)
                    tpl_macro = self.get_scaled_rotated_template(ref_macro, scale, angle)

                    th, tw = tpl_dog.shape
                    if th >= sh or tw >= sw:
                        continue

                    # Compute ZNCC on DoG intensity map
                    res_dog  = cv2.matchTemplate(search_dog,   tpl_dog,  cv2.TM_CCOEFF_NORMED)
                    # Compute ZNCC on Sobel edge magnitude map
                    res_edge = cv2.matchTemplate(search_edge,  tpl_edge, cv2.TM_CCOEFF_NORMED)
                    # Compute ZNCC on Raw intensity map (preserves low-frequency macro envelope)
                    tpl_raw  = self.get_scaled_rotated_template(reference_img, scale, angle)
                    res_raw  = cv2.matchTemplate(search_img,   tpl_raw,  cv2.TM_CCOEFF_NORMED)
                    # Compute ZNCC on Macro Structural Envelope
                    res_macro_ch = cv2.matchTemplate(search_macro, tpl_macro, cv2.TM_CCOEFF_NORMED)

                    # Multi-Feature Evidence Fusion Score (DoG + Edge + Raw + Macro Structural Envelope)
                    res_combined = 0.35 * res_dog + 0.25 * res_edge + 0.20 * res_raw + 0.20 * res_macro_ch
                    max_val = float(res_combined.max())

                    # Spatial local correlation peak detection
                    # Threshold at 85% (was 90%) to surface more periodic alias candidates
                    # for AmbiguityResolver to apply closest-to-center tie-breaking
                    kernel = np.ones((9, 9), np.uint8)
                    dilated = cv2.dilate(res_combined, kernel)
                    local_max_mask = (
                        (res_combined == dilated)
                        & (res_combined >= max_val * 0.85)
                        & (res_combined > 0.05)
                    )

                    peak_coords = np.argwhere(local_max_mask)

                    for py, px in peak_coords:
                        score_val = float(res_combined[py, px])
                        center_x = px + tw / 2.0
                        center_y = py + th / 2.0

                        fine_candidates.append({
                            "center_x":    float(center_x),
                            "center_y":    float(center_y),
                            "top_left_x":  int(px),
                            "top_left_y":  int(py),
                            "template_w":  tw,
                            "template_h":  th,
                            "scale_ratio": float(scale),
                            "rotation_deg": float(angle),
                            "score_combined": score_val,
                            "score_dog":   float(res_dog[py, px]),
                            "score_edge":  float(res_edge[py, px]),
                            "score_macro": float(res_macro_ch[py, px]),
                            "response_map": res_combined
                        })

        fine_candidates.sort(key=lambda c: c["score_combined"], reverse=True)
        return fine_candidates[:top_k]
