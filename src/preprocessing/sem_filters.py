import numpy as np
import cv2

class SEMPreprocessor:
    """
    SEM Image Preprocessor extracting high-frequency edge transitions
    and suppressing low-frequency surface charging variations.
    """

    @staticmethod
    def difference_of_gaussians(image: np.ndarray, sigma1: float = 1.0, sigma2: float = 3.0) -> np.ndarray:
        """
        Applies Difference-of-Gaussians (DoG) filter to isolate wafer feature boundaries.
        """
        g1 = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma1)
        g2 = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma2)
        dog = g1 - g2
        # Normalize to 0-255 range
        norm = cv2.normalize(dog, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  # type: ignore[call-overload]
        return norm.astype(np.uint8)

    @staticmethod
    def sobel_gradient_magnitude(image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Computes Sobel gradient magnitude image map.
        """
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)
        mag = cv2.magnitude(gx, gy)
        norm = cv2.normalize(mag, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  # type: ignore[call-overload]
        return norm.astype(np.uint8)

    @staticmethod
    def normalize_intensity(image: np.ndarray) -> np.ndarray:
        """
        Performs zero-mean unit-variance intensity standardization.
        """
        img_f = image.astype(np.float32)
        mean, std = np.mean(img_f), np.std(img_f)
        if std < 1e-6:
            return np.zeros_like(image, dtype=np.float32)
        return (img_f - mean) / std

    @staticmethod
    def macro_envelope(image: np.ndarray, sigma: float = 15.0) -> np.ndarray:
        """
        Applies heavy Gaussian smoothing to blur out high-frequency periodic contacts
        and isolate unique macro wafer structures (scribe lines, peripheral logic, boundaries).
        """
        blur = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)
        norm = cv2.normalize(blur, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  # type: ignore[call-overload]
        return norm.astype(np.uint8)

    def preprocess(self, image: np.ndarray, method: str = "dog") -> np.ndarray:
        """
        Executes configured preprocessing filter.
        """
        if method == "dog":
            return self.difference_of_gaussians(image)
        elif method == "sobel":
            return self.sobel_gradient_magnitude(image)
        elif method == "macro":
            return self.macro_envelope(image)
        elif method == "raw":
            return image.copy()
        else:
            return self.difference_of_gaussians(image)
