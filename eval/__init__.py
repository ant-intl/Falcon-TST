"""
Test suite for Falcon-TST

This package contains comprehensive tests for the Falcon-TST foundation model,
including unit tests, integration tests, and performance benchmarks.
"""
from .evaluation import  Eval
from .metrics import metric
__version__ = "0.1.0"
__all__ = [
    "Eval",
    "metric"
]