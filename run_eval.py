"""
PatchMoE Benchmark Evaluation Example

This script demonstrates how to evaluate a pre-trained PatchMoE model on multiple
benchmark datasets including ETT (Electricity Transforming Temperature), Weather,
and Electricity datasets. It loads a model checkpoint and runs comprehensive
evaluation across all specified datasets.
"""

import torch
from eval import Eval
from transformers import AutoModelForCausalLM
import argparse


if __name__ == "__main__":
    """
    Main evaluation script for PatchMoE model benchmarking.

    This script:
    1. Configures evaluation parameters for multiple datasets
    2. Loads a pre-trained PatchMoE model from checkpoint
    3. Runs evaluation across all benchmark datasets
    4. Saves results and prints performance summary
    """

    # Disable gradient computation for evaluation
    with torch.no_grad():
        parser = argparse.ArgumentParser(description="Evaluation")
        parser.add_argument("--root_path", type=str, default="./dataset/", help="Root path of the datasets.")
        parser.add_argument("--ckpt_path", type=str, default="", help="Checkpoint path of the model.")
        parser.add_argument("--output_path", type=str, default="./results/", help="Output path of the results.")
        parser.add_argument("--test_data_list", type=str, nargs="+", default=[], help="List of datasets to evaluate.")
        parser.add_argument("--device", type=str, default='cuda:0', help="Model device.")
        parser.add_argument("--seq_length", type=int, default=2880, help="Input length of time series.")
        parser.add_argument("--pred_length", type=int, default=96, help="Output length of time series.")
        parser.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size.")
        args = parser.parse_args()

        if not torch.cuda.is_available():
            args.device = 'cpu'

        # Initialize evaluator with configuration
        evaluator = Eval(args)

        # Load pre-trained PatchMoE model from checkpoint
        print(f"Loading model from: {args.ckpt_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path,
            trust_remote_code=True,
            local_files_only=True
        ).to(device=args.device)
        
        # Run comprehensive evaluation across all datasets
        print("Starting benchmark evaluation...")
        eval_res = evaluator.test(model)

        # Display final results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 50)
        print(eval_res)
