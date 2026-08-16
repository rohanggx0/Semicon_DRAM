import os, sys
# pyrefly: ignore [missing-import]
import numpy as np
import cv2
from scipy.fft import fft2 as sfft2, ifft2 as sifft2
from typing import List, Dict, Any, Tuple

from src.preprocessing.sem_filters import SEMPreprocessor


class PhaseCorrelationMatcher:
    """
    Phase Correlation based scale-aware SEM wafer target localizer.

    Physical reasoning:
    -------------------
    ZNCC template matching on periodic DRAM arrays produces identical
    local maxima at every unit-cell repeat (pitch aliasing). Phase
    Correlation works in the Fourier domain and finds the GLOBAL
    translation peak rather than one of many local periodic peaks.

    For each candidate scale ratio S in [scale_min, scale_max]:
      1. Resize 1000x1000 reference to (1000/S) x (1000/S) pixels.
      2. Embed the resized reference template into a zero-padded
         1000x1000 canvas (keeps FFT sizes uniform and avoids
         circular convolution artifacts at borders).
      3. Compute the normalised cross-power spectrum between
         the padded template and the search image and take its
         inverse FFT.  The translation peak gives (tx, ty).
      4. The predicted target centre in search coordinates is:
             x = tx + (template_w / 2)
             y = ty + (template_h / 2)
      5. Also run on the Sobel edge map to add a second evidence
         channel; fuse scores to form a combined confidence.

    Scale search: S is sampled at scale_step intervals.
    Rotation   : small rotations (<=2 deg) are handled by rotating
                 the resized template before embedding.

    Reference:
        Foroosh, H., Zerubia, J. B., & Berthod, M. (2002).
        Extension of phase correlation to subpixel registration.
        IEEE TIP.
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
        self.scale_range  = np.arange(scale_min,  scale_max  + 1e-5, scale_step)
        self.rotation_range = np.arange(rotation_min, rotation_max + 1e-5, rotation_step)
        self.preprocessor = SEMPreprocessor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fft2_search(img: np.ndarray) -> np.ndarray:
        """
        Pre-compute FFT2 of a search image for reuse across all template iterations.
        No windowing is applied: the search image covers the full 1000x1000 FOV and
        windowing would bias correlation towards the image centre.
        """
        return sfft2(img.astype(np.float32), workers=-1)

    @staticmethod
    def _phase_correlation_peak(
        F_img: np.ndarray,
        tpl: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute normalised cross-power spectrum and return
        (tx, ty, peak_value) where (tx, ty) is the best integer
        translation of tpl inside img.

        Parameters
        ----------
        F_img : pre-computed FFT2 of the H x W float32 search image.
        tpl   : H x W float32 template — same size as img, zero-padded.

        Returns
        -------
        tx, ty : integer pixel shift of template top-left corner.
        peak_val : normalised correlation peak height in [0, 1].
        """
        H, W = tpl.shape

        # FFT of template only (search image FFT is pre-computed).
        # No Hanning window on the template: the template content is at the
        # top-left corner of the zero-padded canvas, where a full-canvas
        # Hanning window would be ≈0, destroying the signal.
        # The zero-padding itself acts as a natural boundary taper.
        F_tpl = sfft2(tpl.astype(np.float32), workers=-1)

        # Cross-correlation (unnormalized) for template matching.
        # NOTE: We intentionally do NOT use the phase-only normalization
        # (R = cross / |cross|) here. That normalization is only valid for
        # whole-image shift estimation between two identically-sized images
        # of the same content. For zero-padded template-in-image matching,
        # the unnormalized cross-correlation correctly localizes the template.
        cross = F_img * np.conj(F_tpl)

        # Inverse FFT → cross-correlation surface
        r = np.abs(sifft2(cross, workers=-1)).astype(np.float32)

        # Find peak location using normalized surface for stability
        r_max  = float(r.max()) + 1e-10
        r_norm = r / r_max
        _, _, _, max_loc = cv2.minMaxLoc(r_norm)
        tx, ty = max_loc   # OpenCV gives (col, row) → that is (tx, ty)

        # Compute Peak-to-Sidelobe Ratio (PSR) as the quality score.
        # PSR measures how sharply the true peak stands out relative to
        # background correlation noise — enables meaningful ranking across
        # different scale/rotation hypotheses (unlike per-surface normalization
        # which would make every candidate score = 1.0).
        py, px = ty, tx
        mask = np.ones(r.shape, dtype=bool)
        y0, y1 = max(0, py - 3), min(H, py + 4)
        x0, x1 = max(0, px - 3), min(W, px + 4)
        mask[y0:y1, x0:x1] = False
        sidelobe_vals = r[mask]
        sl_mean = float(sidelobe_vals.mean()) + 1e-10
        sl_std  = float(sidelobe_vals.std())  + 1e-10
        psr = (r_max - sl_mean) / sl_std

        # The phase-correlation surface wraps around; a peak near the
        # right or bottom edge means a negative shift.  Unwrap:
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
        """
        Resize the 1000x1000 reference to (template_size x template_size)
        to match the expected footprint in the search image, optionally
        rotate it, then zero-pad to (target_size x target_size) so that
        phase-correlation FFT sizes are uniform.

        Returns
        -------
        padded : target_size x target_size float32 zero-padded template.
        tw     : un-padded template width in pixels.
        th     : un-padded template height in pixels.
        """
        # Expected size of the reference in search-image pixels
        tsize = max(10, min(target_size - 2, int(round(float(target_size) / scale))))

        # Downsample reference to the expected template size
        small_tpl = cv2.resize(ref_img.astype(np.float32), (tsize, tsize),
                               interpolation=cv2.INTER_AREA)

        # Optional small rotation
        if abs(angle) > 1e-4:
            M = cv2.getRotationMatrix2D((tsize / 2.0, tsize / 2.0), angle, 1.0)
            small_tpl = cv2.warpAffine(small_tpl, M, (tsize, tsize),
                                       borderMode=cv2.BORDER_REFLECT)

        # Zero-pad to target_size x target_size (place template at top-left)
        padded = np.zeros((target_size, target_size), dtype=np.float32)
        padded[:tsize, :tsize] = small_tpl

        return padded, tsize, tsize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_candidates(
        self,
        reference_img: np.ndarray,
        search_img: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Run phase-correlation search over scale × rotation grid.
        Returns top_k candidate dictionaries sorted by combined score.

        Each candidate dict contains:
          center_x, center_y    : predicted target centre in search coords
          top_left_x, top_left_y: template top-left in search coords
          template_w, template_h : scaled template dimensions
          scale_ratio            : the scale S that produced this candidate
          rotation_deg           : the rotation angle used
          score_combined         : fused evidence score
          score_dog, score_edge  : per-channel scores
          response_map           : None (phase correlation doesn't return a map)
        """
        H, W = search_img.shape
        assert H == W == 1000, "Search image must be 1000x1000"

        # Preprocess both images: DoG (structure) and Sobel (edges)
        ref_dog    = self.preprocessor.preprocess(reference_img, method="dog").astype(np.float32)
        search_dog = self.preprocessor.preprocess(search_img,    method="dog").astype(np.float32)

        ref_edge    = self.preprocessor.preprocess(reference_img, method="sobel").astype(np.float32)
        search_edge = self.preprocessor.preprocess(search_img,    method="sobel").astype(np.float32)

        all_candidates: List[Dict[str, Any]] = []

        # Pre-compute FFTs of search images once — reused for every scale/rotation iteration
        F_search_dog  = self._fft2_search(search_dog)
        F_search_edge = self._fft2_search(search_edge)

        for scale in self.scale_range:
            for angle in self.rotation_range:
                # --- DoG channel ---
                tpl_dog, tw, th = self._make_padded_template(ref_dog, scale, angle, 1000)
                tx_dog, ty_dog, peak_dog = self._phase_correlation_peak(F_search_dog, tpl_dog)

                # --- Edge channel ---
                tpl_edge, _, _ = self._make_padded_template(ref_edge, scale, angle, 1000)
                tx_edge, ty_edge, peak_edge = self._phase_correlation_peak(F_search_edge, tpl_edge)

                # Fuse: weighted average of shifts and scores
                score_combined = 0.5 * peak_dog + 0.5 * peak_edge

                # Use DoG shift as primary (edges can alias differently)
                tx = tx_dog
                ty = ty_dog

                # The phase correlation peak at (tx, ty) means the template
                # top-left is at (tx, ty) in the search image.
                # Centre of the target in search coords:
                center_x = float(tx + tw / 2.0)
                center_y = float(ty + th / 2.0)

                # Reject candidates whose centres fall outside the search image
                if not (0 <= center_x < W and 0 <= center_y < H):
                    continue

                all_candidates.append({
                    "center_x":     center_x,
                    "center_y":     center_y,
                    "top_left_x":   int(tx),
                    "top_left_y":   int(ty),
                    "template_w":   tw,
                    "template_h":   th,
                    "scale_ratio":  float(scale),
                    "rotation_deg": float(angle),
                    "score_combined": float(score_combined),
                    "score_dog":    float(peak_dog),
                    "score_edge":   float(peak_edge),
                    "response_map": None,   # not available for phase correlation
                })

        all_candidates.sort(key=lambda c: c["score_combined"], reverse=True)
        return all_candidates[:top_k]
