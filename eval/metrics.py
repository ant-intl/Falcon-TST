"""
Metrics Module for Time Series Forecasting Evaluation

This module provides standard evaluation metrics for time series forecasting tasks,
including Mean Absolute Error (MAE) and Mean Squared Error (MSE).
"""

import numpy as np


def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error between predictions and ground truth.
    
    MAE measures the average magnitude of errors in predictions, without considering
    their direction. It's a linear score which means all individual differences
    are weighted equally in the average.
    
    Args:
        pred (np.ndarray): Predicted values
        true (np.ndarray): Ground truth values
        
    Returns:
        float: Mean Absolute Error value
    """
    return np.mean(np.abs(true - pred))


def MSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between predictions and ground truth.
    
    MSE measures the average of the squares of the errors. It gives higher weight
    to larger errors compared to MAE, making it more sensitive to outliers.
    
    Args:
        pred (np.ndarray): Predicted values
        true (np.ndarray): Ground truth values
        
    Returns:
        float: Mean Squared Error value
    """
    return np.mean((true - pred) ** 2)


def metric(pred: np.ndarray, true: np.ndarray) -> tuple:
    """
    Calculate both MAE and MSE metrics for time series forecasting evaluation.
    
    This is a convenience function that computes both primary evaluation metrics
    used in time series forecasting benchmarks.
    
    Args:
        pred (np.ndarray): Predicted values of shape [batch_size, seq_len, features]
        true (np.ndarray): Ground truth values of shape [batch_size, seq_len, features]
        
    Returns:
        tuple: A tuple containing (mae, mse) values
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    
    return mae, mse
