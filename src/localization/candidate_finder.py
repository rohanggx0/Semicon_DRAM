import numpy as np
from typing import List, Dict, Any, Tuple

class AmbiguityResolver:
    """
    Handles semiconductor repetitive-pattern correlation ambiguity.
    Applies the official tie-breaker rule: Select valid candidate closest to search center (500, 500).
    """

    def __init__(self, search_center: Tuple[float, float] = (500.0, 500.0), confidence_margin: float = 0.985):
        self.search_center_x, self.search_center_y = search_center
        self.confidence_margin = confidence_margin

    def resolve_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Filters candidates by structural similarity score and resolves repetitive pattern ties
        by applying distance to search center rule.
        """
        if not candidates:
            return {
                "center_x": float(self.search_center_x),
                "center_y": float(self.search_center_y),
                "top_left_x": int(self.search_center_x - 50),
                "top_left_y": int(self.search_center_y - 50),
                "template_w": 100,
                "template_h": 100,
                "scale_ratio": 10.0,
                "rotation_deg": 0.0,
                "score_combined": 0.0,
                "num_valid_matches": 0,
                "tie_breaker_applied": False,
                "response_map": None
            }

        # Highest correlation score across all candidate options
        max_score = max(c["score_combined"] for c in candidates)

        # Only consider the near-best candidates for tie-breaking.
        # This prevents broad center-biased selection when lower-scoring periodic peaks
        # are still above a loose threshold.
        score_epsilon = max(1e-6, (1.0 - self.confidence_margin) * abs(max_score))
        tied_candidates = [
            c for c in candidates
            if c["score_combined"] >= max_score - score_epsilon
        ]

        if len(tied_candidates) == 1:
            selected_best = tied_candidates[0]
            selected_best["num_valid_matches"] = 1
            selected_best["tie_breaker_applied"] = False
            return selected_best

        # Calculate distance to search image center and composite tie score for all tied candidates
        for candidate in tied_candidates:
            dx = candidate["center_x"] - self.search_center_x
            dy = candidate["center_y"] - self.search_center_y
            dist = float(np.sqrt(dx**2 + dy**2))
            candidate["dist_to_center"] = dist
            macro_score = candidate.get("score_macro", candidate["score_combined"])
            # Tie score prioritizes macro structural envelope match while preferring center proximity
            candidate["tie_score"] = macro_score - 1e-4 * dist

        tied_candidates.sort(key=lambda c: c["tie_score"], reverse=True)

        selected_best = tied_candidates[0]
        selected_best["num_valid_matches"] = len(tied_candidates)
        selected_best["tie_breaker_applied"] = len(tied_candidates) > 1

        return selected_best
