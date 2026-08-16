"""
DriftSense AI Model Registry.

Models:
  SEMRestorationUNet  – Localization-Aware Edge-Preserving U-Net for SEM image denoising
  AIRestorationModel  – Alias for SEMRestorationUNet (used by drift_sense pipeline)
  DoubleConv          – Basic (Conv → BN → ReLU) × 2 building block
"""

from models.ai_restoration import SEMRestorationUNet, DoubleConv

# Alias used by the Drift-Sense pipeline config (ai.model = AIRestorationModel)
AIRestorationModel = SEMRestorationUNet

__all__ = ["SEMRestorationUNet", "AIRestorationModel", "DoubleConv"]
