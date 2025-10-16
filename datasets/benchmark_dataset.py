#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Benchmark Dataset Module for Falcon-TST Time Series Evaluation

This module provides dataset classes for loading and preprocessing time series data
from various benchmark datasets (ETT, Weather, Electricity, etc.) for evaluation
of the Falcon-TST model.
"""

import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class BenchmarkEvalDataset(Dataset):
    """
    Dataset class for benchmark time series evaluation.

    This class handles loading, preprocessing, and serving time series data from
    various benchmark datasets including ETT (Electricity Transforming Temperature),
    Weather, and Electricity datasets. It automatically handles data splitting,
    normalization, and windowing for time series forecasting evaluation.

    Args:
        csv_path (str): Path to the CSV file containing the time series data
        context_length (int): Length of the input context window (lookback period)
        prediction_length (int): Length of the prediction horizon (forecast period)

    Attributes:
        context_length (int): Input sequence length for the model
        prediction_length (int): Output sequence length for forecasting
        window_length (int): Total window length (context + prediction)
        num_sequences (int): Number of time series sequences in the dataset
        scaler_list (list): List of StandardScaler objects for each sequence
        sub_seq_indexes (list): List of (sequence_idx, offset_idx) tuples for sampling
        hf_dataset (np.ndarray): Preprocessed and normalized test data
    """

    def __init__(self, csv_path: str, context_length: int, prediction_length: int):
        """
        Initialize the BenchmarkEvalDataset.

        Args:
            csv_path (str): Path to the CSV file containing time series data
            context_length (int): Length of input context window
            prediction_length (int): Length of prediction horizon
        """
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_length = self.context_length + self.prediction_length

        # Load the CSV data
        df = pd.read_csv(csv_path)

        # Determine dataset-specific data splitting boundaries
        base_name = os.path.basename(csv_path).lower()
        if "etth" in base_name:
            # ETT hourly datasets: 12 months train, 4 months validation, 4 months test
            border1s = [
                0,
                12 * 30 * 24 - context_length,
                12 * 30 * 24 + 4 * 30 * 24 - context_length,
            ]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif "ettm" in base_name:
            # ETT minute datasets: 12 months train, 4 months validation, 4 months test (4x more data points)
            border1s = [
                0,
                12 * 30 * 24 * 4 - context_length,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - context_length,
            ]
            border2s = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]
        else:
            # Other datasets: 70% train, 20% validation, 20% test split
            num_train = int(len(df) * 0.7)
            num_test = int(len(df) * 0.2)
            num_vali = len(df) - num_train - num_test
            border1s = [0, num_train - context_length, len(df) - num_test - context_length]
            border2s = [num_train, num_train + num_vali, len(df)]

        # Log the test data time range for debugging
        start_dt = df.iloc[border1s[2]]["date"]
        eval_start_dt = df.iloc[border1s[2] + context_length]["date"]
        end_dt = df.iloc[border2s[2] - 1]["date"]
        print(
            f">>> Split test data from {start_dt} to {end_dt}, "
            f"and evaluation start date is: {eval_start_dt}"
        )

        # Extract feature columns (exclude date column)
        cols = df.columns[1:]
        df_values = df[cols].values

        # Split data into train and test sets
        train_data = df_values[border1s[0] : border2s[0]].transpose(
            1, 0
        )  # Shape: [num_features, train_length]
        test_data = df_values[border1s[2] : border2s[2]].transpose(
            1, 0
        )  # Shape: [num_features, test_length]

        # Initialize dataset attributes
        self.num_sequences = len(train_data)
        self.scaler_list = []
        self.sub_seq_indexes = []

        # Process each time series sequence
        for idx in range(self.num_sequences):
            train_seq = train_data[idx].reshape(-1, 1)
            test_seq = test_data[idx].reshape(-1, 1)
            n_points = len(test_seq)

            # Skip sequences that are too short for windowing
            if n_points < self.window_length:
                continue

            # Create sliding window indices for this sequence
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((idx, offset_idx))

            # Fit StandardScaler on training data and transform test data
            scaler = StandardScaler()
            scaler.fit(train_seq)
            scaled_test_seq = scaler.transform(test_seq)
            test_data[idx] = scaled_test_seq[:, 0]
            self.scaler_list.append(scaler)

        self.hf_dataset = test_data

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.sub_seq_indexes)

    def __iter__(self):
        """Make the dataset iterable."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: Dictionary containing:
                - 'inputs': Input context sequence of shape [context_length]
                - 'labels': Target prediction sequence of shape [prediction_length]
                - 'seq_idx': Index of the source time series sequence
        """
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.hf_dataset[seq_i]

        # Extract the windowed sequence
        window_seq = np.array(seq[offset_i - self.window_length : offset_i], dtype=np.float32)
        assert len(window_seq) == self.window_length

        return {
            "inputs": np.array(window_seq[: self.context_length], dtype=np.float32),
            "labels": np.array(window_seq[-self.prediction_length :], dtype=np.float32),
            "seq_idx": seq_i,
        }

    def inverse_transform(self, data: np.ndarray, seq_idx) -> np.ndarray:
        """
        Apply inverse transformation to denormalize predictions.

        This method reverses the StandardScaler normalization applied during
        preprocessing to convert predictions back to the original scale.

        Args:
            data (np.ndarray): Normalized prediction data of shape [batch_size, pred_length]
            seq_idx (int or np.ndarray): Sequence index(es) corresponding to the data

        Returns:
            np.ndarray: Denormalized prediction data in original scale
        """
        if type(seq_idx) == int:
            # Single sequence case
            return self.scaler_list[seq_idx].inverse_transform(data)
        else:
            # Batch case - apply inverse transform for each sample
            inversed_data = data.copy()
            for i in range(data.shape[0]):
                inversed_data[i] = self.scaler_list[seq_idx[i]].inverse_transform(
                    data[i].reshape(-1, 1)
                )[:, 0]
        return inversed_data
