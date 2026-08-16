import numpy as np
from typing import Dict, Any, Tuple

class SubpixelRefiner:
    """
    Sub-pixel peak refinement engine fitting a 2D quadratic / parabolic
    correlation surface around integer correlation maximum.
    """

    def __init__(self, method: str = "quadratic"):
        self.method = method

    def refine_peak(self, candidate: Dict[str, Any]) -> Tuple[float, float]:
        """
        Refines integer target center candidate (x, y) using local correlation surface geometry.
        """
        x_int = candidate["center_x"]
        y_int = candidate["center_y"]
        
        top_left_x = candidate["top_left_x"]
        top_left_y = candidate["top_left_y"]
        res_map = candidate.get("response_map", None)

        if res_map is None:
            return x_int, y_int

        h, w = res_map.shape

        # Ensure 3x3 neighborhood is within response map boundaries
        if top_left_x < 1 or top_left_x >= w - 1 or top_left_y < 1 or top_left_y >= h - 1:
            return x_int, y_int

        # Extract 3x3 neighborhood values
        r_center = res_map[top_left_y, top_left_x]
        r_left   = res_map[top_left_y, top_left_x - 1]
        r_right  = res_map[top_left_y, top_left_x + 1]
        r_top    = res_map[top_left_y - 1, top_left_x]
        r_bottom = res_map[top_left_y + 1, top_left_x]

        # Quadratic peak shift calculation along X axis
        denom_x = r_left - 2.0 * r_center + r_right
        if abs(denom_x) > 1e-6:
            delta_x = (r_left - r_right) / (2.0 * denom_x)
        else:
            delta_x = 0.0

        # Quadratic peak shift calculation along Y axis
        denom_y = r_top - 2.0 * r_center + r_bottom
        if abs(denom_y) > 1e-6:
            delta_y = (r_top - r_bottom) / (2.0 * denom_y)
        else:
            delta_y = 0.0

        # Constrain sub-pixel shift to valid [-0.5, 0.5] pixel offset interval
        delta_x = np.clip(delta_x, -0.5, 0.5)
        delta_y = np.clip(delta_y, -0.5, 0.5)

        x_refined = x_int + delta_x
        y_refined = y_int + delta_y

        return float(x_refined), float(y_refined)
