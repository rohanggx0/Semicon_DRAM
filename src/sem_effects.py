"""
Physics-Based Scanning Electron Microscope (SEM) Degradation Pipeline — V1.5 Anisotropic Optics

Implements literature-supported SEM image formation models:
1. Secondary Electron (SE) Edge Emission Bloom
2. Anisotropic Electron Beam Astigmatic PSF Blur
3. Poisson Primary/Secondary Electron Shot Noise
4. Additive Gaussian Detector/Readout Noise
5. Wafer Surface Charging Streak Artifacts
6. Photomultiplier Contrast & Gamma Dynamic Range Shifts
7. Mechanical Stage/Scanning Relative Rotation Variances
"""

import math
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image


def apply_edge_brightening(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Simulate Secondary Electron (SE) edge emission bloom.
    Edges emit higher secondary electrons due to larger escape depth at tilted surfaces.
    """
    if strength <= 0.0:
        return img

    f_img = img.astype(np.float32)
    gx = cv2.Sobel(f_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f_img, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(gx, gy)
    max_v = float(edge_mag.max())
    if max_v > 1e-4:
        edge_mag = (edge_mag / max_v) * 255.0

    out = np.clip(f_img + strength * edge_mag, 0, 255).astype(np.uint8)
    return out


def apply_gaussian_blur(
    img: np.ndarray,
    sigma: float = 1.0,
    astigmatism_ratio: float = 1.25,
    theta_deg: float = 30.0
) -> np.ndarray:
    """
    Simulate anisotropic electron beam finite spot-size Gaussian Point Spread Function (PSF).
    Models electron beam astigmatism where sigma_x != sigma_y along beam astigmatism angle theta_deg:
    G(x', y') = exp(-0.5 * (x'^2 / sigma_x^2 + y'^2 / sigma_y^2))
    """
    if sigma <= 0.0:
        return img

    sigma_x = sigma
    sigma_y = sigma * astigmatism_ratio

    ks_x = max(3, int(round(6 * sigma_x)) | 1)
    ks_y = max(3, int(round(6 * sigma_y)) | 1)

    ax = np.arange(-ks_x // 2 + 1, ks_x // 2 + 1)
    ay = np.arange(-ks_y // 2 + 1, ks_y // 2 + 1)
    xx, yy = np.meshgrid(ax, ay)

    rad = math.radians(theta_deg)
    xr = xx * math.cos(rad) + yy * math.sin(rad)
    yr = -xx * math.sin(rad) + yy * math.cos(rad)

    kernel = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / (sigma_y + 1e-6)) ** 2))
    kernel /= kernel.sum()

    blurred = cv2.filter2D(img, -1, kernel)
    return blurred.astype(np.uint8)


def apply_poisson_shot_noise(img: np.ndarray, dose_factor: float = 100.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Simulate electron shot noise based on Poisson counting statistics of arriving primary/secondary electrons.
    Lower dose_factor corresponds to lower electron dose (noisier image).
    """
    if dose_factor <= 0.0:
        return img

    rng = np.random.default_rng(seed)
    f_img = img.astype(np.float32) / 255.0
    lam = np.maximum(f_img * dose_factor, 1e-4)

    noisy_counts = rng.poisson(lam).astype(np.float32)
    noisy_img = noisy_counts / dose_factor
    noisy_img = np.clip(noisy_img * 255.0, 0, 255).astype(np.uint8)

    return noisy_img


def apply_gaussian_detector_noise(img: np.ndarray, std_dev: float = 5.0, seed: Optional[int] = None) -> np.ndarray:
    """Simulate thermal and readout amplifier Gaussian noise in SEM detector circuit."""
    if std_dev <= 0.0:
        return img

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std_dev, size=img.shape).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return noisy_img


def apply_surface_charging_streaks(img: np.ndarray, strength: float = 15.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Simulate dielectric surface charging artifacts.
    Low-conductivity oxides/resists accumulate charge, deflecting electron beam horizontally.
    """
    if strength <= 0.0:
        return img

    rng = np.random.default_rng(seed)
    h, w = img.shape

    num_streaks = rng.integers(3, 8)
    streak_mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_streaks):
        y_start = rng.integers(0, h - 10)
        streak_len = rng.integers(10, 40)
        intensity = rng.uniform(0.5, 1.0) * strength

        y_end = min(h, y_start + streak_len)
        streak_mask[y_start:y_end, :] += intensity

    out = np.clip(img.astype(np.float32) + streak_mask, 0, 255).astype(np.uint8)
    return out


def apply_contrast_and_gamma(img: np.ndarray, contrast: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Simulate SEM photomultiplier dynamic range scaling and non-linear detector gamma response."""
    f_img = img.astype(np.float32) / 255.0

    if abs(contrast - 1.0) > 1e-3:
        f_img = (f_img - 0.5) * contrast + 0.5

    if abs(gamma - 1.0) > 1e-3 and gamma > 0:
        f_img = np.power(np.maximum(f_img, 0.0), gamma)

    out = np.clip(f_img * 255.0, 0, 255).astype(np.uint8)
    return out


def apply_rotation(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply mechanical stage / electron optical scanning rotation variance."""
    if abs(angle_deg) <= 1e-4:
        return img

    h, w = img.shape[:2]
    center = (w / 2.0 - 0.5, h / 2.0 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_full_sem_pipeline(
    raw_img: np.ndarray,
    edge_bloom: float = 0.3,
    blur_sigma: float = 0.8,
    shot_dose: float = 150.0,
    gaussian_std: float = 6.0,
    charging_strength: float = 10.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    rotation_deg: float = 0.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Complete literature-grounded SEM transfer function pipeline.
    Sequentially applies rotation, SE edge bloom, anisotropic PSF blur, contrast/gamma, charging,
    Poisson shot noise, and Gaussian detector noise.
    """
    s_base = seed if seed is not None else 42

    x = apply_rotation(raw_img, rotation_deg)
    x = apply_edge_brightening(x, strength=edge_bloom)
    x = apply_gaussian_blur(x, sigma=blur_sigma, astigmatism_ratio=1.25, theta_deg=30.0)
    x = apply_contrast_and_gamma(x, contrast=contrast, gamma=gamma)
    x = apply_surface_charging_streaks(x, strength=charging_strength, seed=s_base + 1)
    x = apply_poisson_shot_noise(x, dose_factor=shot_dose, seed=s_base + 2)
    x = apply_gaussian_detector_noise(x, std_dev=gaussian_std, seed=s_base + 3)

    return x
