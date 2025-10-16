"""
Falcon-TST: A Large-Scale Time Series Foundation Model with Mixture of Experts Architecture

This package provides a comprehensive implementation of Falcon-TST, a time series foundation model
that utilizes Mixture of Experts (MoE) architecture with multiple patch tokenizers for
efficient and accurate time series forecasting.
"""

__version__ = "0.1.0"
__author__ = "Falcon-TST Team"
__email__ = "falcontst@example.com"

# Import main classes for easy access
from .models.configuration_falcon_tst import FalconTSTConfig
from .models.modeling_falcon_tst import (
    FalconTSTForPrediction,
    FalconTSTModel,
    FalconTSTPreTrainedModel,
)

# Define what should be imported with "from falcon_tst import *"
__all__ = [
    "FalconTSTConfig",
    "FalconTSTForPrediction",
    "FalconTSTModel",
    "FalconTSTPreTrainedModel",
    "__version__",
]
