"""
Phase Correlation based scale-aware SEM wafer target localizer.

Finds translation peaks in the Fourier domain across scale and rotation search grids.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from scipy.fft import fft2 as sfft2, ifft2 as sifft2


class PhaseCorrelationMatcher:
    """
    Fourier Phase Correlation Matcher across Scale and Rotation.
    """

    def __init__(
        self,
        scale_min: float = 8.5,
        scale_max: float = 11.5,
        scale_step: float = 0.25,
        rotation_min: float = -2.5,
        rotation_max: float = 2.5,
        rotation_step: float = 0.5,
    ):
        self.scale_range = np.arange(scale_min, scale_max + 1e-5, scale_step)
        self.rotation_range = np.arange(rotation_min, rotation_max + 1e-5, rotation_step)

    @staticmethod
    def _fft2_search(img: np.ndarray) -> np.ndarray:
        return sfft2(img.astype(np.float32), workers=-1)

    @staticmethod
    def _phase_correlation_peak(F_img: np.ndarray, tpl: np.ndarray) -> Tuple[float, float, float]:
        H, W = tpl.shape
        F_tpl = sfft2(tpl.astype(np.float32), workers=-1)
        cross = F_img * np.conj(F_tpl)
        r = np.abs(sifft2(cross, workers=-1)).astype(np.float32)

        r_max = float(r.max()) + 1e-10
        r_norm = r / r_max
        _, _, _, max_loc = cv2.minMaxLoc(r_norm)
        tx, ty = max_loc

        py, px = ty, tx
        mask = np.ones(r.shape, dtype=bool)
        y0, y1 = max(0, py - 3), min(H, py + 4)
        x0, x1 = max(0, px - 3), min(W, px + 4)
        mask[y0:y1, x0:x1] = False
        sidelobe_vals = r[mask]
        sl_mean = float(sidelobe_vals.mean()) + 1e-10
        sl_std = float(sidelobe_vals.std()) + 1e-10
        psr = (r_max - sl_mean) / sl_std

        if tx > W // 2:
            tx -= W
        if ty > H // 2:
            ty -= H

        return float(tx), float(ty), float(psr)

    def _make_padded_template(
        self,
        ref_img: np.ndarray,
        scale: float,
        angle: float,
        target_size: int = 1000,
    ) -> Tuple[np.ndarray, int, int]:
        tsize = max(10, min(target_size - 2, int(round(float(target_size) / scale))))
        small_tpl = cv2.resize(ref_img.astype(np.float32), (tsize, tsize), interpolation=cv2.INTER_AREA)

        if abs(angle) > 1e-4:
            M = cv2.getRotationMatrix2D((tsize / 2.0, tsize / 2.0), angle, 1.0)
            small_tpl = cv2.warpAffine(small_tpl, M, (tsize, tsize), borderMode=cv2.BORDER_REFLECT)

        padded = np.zeros((target_size, target_size), dtype=np.float32)
        padded[:tsize, :tsize] = small_tpl
        return padded, tsize, tsize

    def search_candidates(
        self,
        reference_img: np.ndarray,
        search_img: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        H, W = search_img.shape

        ref_f32 = reference_img.astype(np.float32)
        search_f32 = search_img.astype(np.float32)

        ref_gx = cv2.Sobel(ref_f32, cv2.CV_32F, 1, 0, ksize=3)
        ref_gy = cv2.Sobel(ref_f32, cv2.CV_32F, 0, 1, ksize=3)
        ref_edge = cv2.magnitude(ref_gx, ref_gy)

        search_gx = cv2.Sobel(search_f32, cv2.CV_32F, 1, 0, ksize=3)
        search_gy = cv2.Sobel(search_f32, cv2.CV_32F, 0, 1, ksize=3)
        search_edge = cv2.magnitude(search_gx, search_gy)

        all_candidates: List[Dict[str, Any]] = []

        F_search_raw = self._fft2_search(search_f32)
        F_search_edge = self._fft2_search(search_edge)

        for scale in self.scale_range:
            for angle in self.rotation_range:
                tpl_raw, tw, th = self._make_padded_template(ref_f32, scale, angle, W)
                tx_raw, ty_raw, psr_raw = self._phase_correlation_peak(F_search_raw, tpl_raw)

                tpl_edge, _, _ = self._make_padded_template(ref_edge, scale, angle, W)
                tx_edge, ty_edge, psr_edge = self._phase_correlation_peak(F_search_edge, tpl_edge)

                combined_score = float(0.5 * psr_raw + 0.5 * psr_edge)

                center_x = float(tx_raw + tw / 2.0)
                center_y = float(ty_raw + th / 2.0)

                if not (0 <= center_x < W and 0 <= center_y < H):
                    continue

                all_candidates.append({
                    "center_x": center_x,
                    "center_y": center_y,
                    "top_left_x": int(tx_raw),
                    "top_left_y": int(ty_raw),
                    "template_w": tw,
                    "template_h": th,
                    "scale_ratio": float(scale),
                    "rotation_deg": float(angle),
                    "score_combined": combined_score,
                    "psr_raw": float(psr_raw),
                    "psr_edge": float(psr_edge),
                })

        all_candidates.sort(key=lambda c: c["score_combined"], reverse=True)
        return all_candidates[:top_k]
