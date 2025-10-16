"""
Evaluation Module for Falcon-TST

This module provides comprehensive evaluation functionality for the Falcon-TST model
on various time series forecasting benchmarks. It handles batch evaluation,
metric computation, and result aggregation across multiple datasets.
"""

import os
import time
import torch
import numpy as np
import pandas as pd


class Eval:
    """
    Evaluation class for Falcon-TST model performance assessment.
    
    This class orchestrates the evaluation process across multiple benchmark datasets,
    computing forecasting metrics and generating comprehensive evaluation reports.
    
    Args:
        args: Configuration object containing evaluation parameters including:
            - output_path: Directory to save evaluation results
            - test_data_list: List of datasets to evaluate
            - seq_length: Input sequence length
            - pred_length: Prediction horizon length
            - batch_size: Batch size for evaluation
    """
    
    def __init__(self, args) -> None:
        """Initialize the evaluation class with configuration parameters."""
        self.args = args

    def batch_eval(self, data_loader, model) -> dict:
        """
        Perform batch evaluation on a single dataset.
        
        This method processes batches of time series data through the model,
        collects predictions and ground truth values, and computes evaluation metrics.
        
        Args:
            data_loader: PyTorch DataLoader containing evaluation samples
            model: Falcon-TST model instance for generating predictions
            
        Returns:
            dict: Dictionary containing computed metrics:
                - 'mae': Mean Absolute Error
                - 'mse': Mean Squared Error
        """
        from eval.metrics import metric

        preds = []
        trues = []
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for _, batches in enumerate(data_loader):
                # Extract inputs and labels from batch
                inputs, labels = batches["inputs"], batches["labels"]
                
                # Generate predictions using the model
                model_output = model.generate(
                    inputs, 
                    max_new_tokens=self.args.pred_length
                )

                # Convert tensors to numpy arrays for metric computation
                model_output = model_output.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                # Collect predictions and ground truth
                preds.append(model_output)
                trues.append(labels)
                
        # Concatenate all batch results
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # Ensure consistent dimensionality for metric computation
        trues = np.expand_dims(trues, axis=-1)
        print("test shape:", preds.shape, trues.shape)
        
        # Reshape for proper metric calculation
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)
        
        # Compute evaluation metrics
        mae, mse = metric(preds, trues)
        metrics_dict = {"mae": mae, "mse": mse}
        return metrics_dict

    def test(self, model):
        """
        Run comprehensive evaluation across multiple benchmark datasets.
        
        This method orchestrates the complete evaluation pipeline:
        1. Sets up evaluation datasets for each benchmark
        2. Runs batch evaluation on each dataset
        3. Collects and aggregates results
        4. Saves individual and combined results to CSV files
        5. Reports performance statistics
        
        Args:
            model: Falcon-TST model instance to evaluate
            
        Returns:
            pd.DataFrame: Aggregated evaluation results across all datasets
        """
        from datasets import BenchmarkEvalDataset
        from torch.utils.data import DataLoader

        # Create output directory if it doesn't exist
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)

        # Initialize tracking variables
        total_samples = 0
        total_test_time = 0.0
        total_res = []
        evaluate_prefix = (f"sl{self.args.seq_length}_"
                          f"pl{self.args.pred_length}")
        
        # Evaluate each benchmark dataset
        for scenario_config in self.args.test_data_list:
            print(f"Evaluating scenario: {scenario_config} ...")
            
            # Create scenario-specific output directory
            res_path = os.path.join(self.args.output_path, 
                                   scenario_config)
            if not os.path.exists(res_path):
                os.makedirs(res_path)

            # Determine dataset file path based on dataset type
            csv_path = None
            if "ETT" in scenario_config:
                csv_path = os.path.join(self.args.root_path, 
                                        "ETT-small", 
                                        f"{scenario_config}.csv")
            else:
                csv_path = os.path.join(self.args.root_path, 
                                        "ETT-small", 
                                        f"{scenario_config}.csv")

            # Initialize dataset and dataloader
            if csv_path is not None:
                test_dataset = BenchmarkEvalDataset(
                    csv_path=csv_path,
                    context_length=self.args.seq_length,
                    prediction_length=self.args.pred_length,
                )
            else:
                raise ValueError("Invalid dataset configuration")

            test_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=self.args.batch_size,
                sampler=None,
                shuffle=False,
                num_workers=2,
                prefetch_factor=2,
                drop_last=False,
            )
            
            total_samples += len(test_dataset)
            print(f"done with setup samples:{len(test_dataset)} ...")

            # Perform evaluation with timing
            scenario_start_time = time.time()
            res = self.batch_eval(test_dataloader, model)
            
            # Add metadata to results
            res["dataset_name"] = scenario_config
            res["evaluate_prefix"] = evaluate_prefix
            
            # Save individual scenario results
            res_df = pd.DataFrame([res])
            res_df.to_csv(os.path.join(res_path, "res.csv"), index=False)
            total_res.append(res_df)

            # Track timing
            scenario_time = time.time() - scenario_start_time
            total_test_time += scenario_time

        # Aggregate and save total results
        total_res_path = os.path.join(self.args.output_path, 
                                     "total_res.csv")
        total_res_df = pd.concat(total_res, ignore_index=True)
        total_res_df.to_csv(total_res_path, index=False)

        # Print evaluation summary
        print(f"Total scenarios evaluated: "
              f"{self.args.test_data_list}")
        print(f"Total samples processed: {total_samples}")
        print(f"Total test time: {total_test_time:.6f} s")
        print(f"Average throughput: "
              f"{total_samples/total_test_time:.2f} samples/s")
        print("All evaluations completed successfully")
        
        return total_res_df
