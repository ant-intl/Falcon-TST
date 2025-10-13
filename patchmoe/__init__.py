"""
PatchMoE: A Large-Scale Time Series Foundation Model with Mixture of Experts Architecture

This package provides a comprehensive implementation of PatchMoE, a time series foundation model
that utilizes Mixture of Experts (MoE) architecture with multiple patch tokenizers for
efficient and accurate time series forecasting.
"""

__version__ = "0.1.0"
__author__ = "PatchMoE Team"
__email__ = "patchmoe@example.com"

# Import main classes for easy access
from .models.configuration_patch_moe import PatchMoeConfig
from .models.modeling_patch_moe import (
    PatchMoEForPrediction,
    PatchMoEModel,
    PatchMoEPreTrainedModel,
)

# Define what should be imported with "from patchmoe import *"
__all__ = [
    "PatchMoeConfig",
    "PatchMoEForPrediction",
    "PatchMoEModel",
    "PatchMoEPreTrainedModel",
    "__version__",
]
