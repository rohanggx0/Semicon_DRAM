import numpy as np
import cv2

class SEMDegradationEngine:
    """
    Physical SEM Image Degradation Engine modeling electron-beam interaction,
    electron statistics, optics PSF, secondary electron detection noise, and raster scan drift.
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def apply_beam_psf_blur(self, image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """
        Models electron beam finite spot size (Gaussian Point Spread Function).
        """
        if sigma <= 0:
            return image.copy()
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        blurred = cv2.GaussianBlur(image.astype(np.float32), (kernel_size, kernel_size), sigma)
        return np.clip(blurred, 0, 255).astype(np.uint8)

    def apply_shot_noise(self, image: np.ndarray, lambda_factor: float = 50.0) -> np.ndarray:
        """
        Models Poisson electron shot noise from secondary electron arrival statistics.
        Lower lambda_factor represents lower beam current / dose (higher shot noise).
        """
        if lambda_factor <= 0:
            return image.copy()
        
        normalized = image.astype(np.float32) / 255.0
        scaled = normalized * lambda_factor
        noisy_scaled = self.rng.poisson(scaled).astype(np.float32)
        noisy = (noisy_scaled / lambda_factor) * 255.0
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_readout_noise(self, image: np.ndarray, std_dev: float = 8.0) -> np.ndarray:
        """
        Models electronic amplification and readout Gaussian noise.
        """
        if std_dev <= 0:
            return image.copy()
        gaussian_noise = self.rng.normal(0, std_dev, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + gaussian_noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_contrast_and_gamma(self, image: np.ndarray, gamma: float = 1.1, contrast: float = 1.0) -> np.ndarray:
        """
        Models surface charging and secondary electron yield variations.
        """
        normalized = image.astype(np.float32) / 255.0
        gamma_corrected = np.power(normalized, gamma)
        contrast_adjusted = (gamma_corrected - 0.5) * contrast + 0.5
        scaled = contrast_adjusted * 255.0
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def apply_raster_jitter(self, image: np.ndarray, max_jitter_px: float = 1.5) -> np.ndarray:
        """
        Models SEM line-by-line raster scan jitter caused by magnetic field ripple or stage drift.
        """
        if max_jitter_px <= 0:
            return image.copy()
        
        h, w = image.shape
        jittered = np.zeros_like(image)
        # Generate row-wise shifts
        row_shifts = self.rng.uniform(-max_jitter_px, max_jitter_px, size=h)
        
        for y in range(h):
            shift = row_shifts[y]
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            jittered[y:y+1, :] = cv2.warpAffine(image[y:y+1, :], M, (w, 1), borderMode=cv2.BORDER_REFLECT)
            
        return jittered

    def degrade_sem_image(
        self,
        image: np.ndarray,
        lambda_shot: float = 50.0,
        readout_std: float = 8.0,
        beam_sigma: float = 1.5,
        gamma: float = 1.1,
        jitter_px: float = 1.0
    ) -> np.ndarray:
        """
        Executes full SEM physical degradation pipeline on input clean image.
        """
        img = self.apply_beam_psf_blur(image, sigma=beam_sigma)
        img = self.apply_contrast_and_gamma(img, gamma=gamma)
        img = self.apply_raster_jitter(img, max_jitter_px=jitter_px)
        img = self.apply_shot_noise(img, lambda_factor=lambda_shot)
        img = self.apply_readout_noise(img, std_dev=readout_std)
        return img
