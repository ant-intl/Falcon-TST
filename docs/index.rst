PatchMoE Documentation
======================

Welcome to PatchMoE's documentation! PatchMoE is a large-scale time series foundation model that utilizes Mixture of Experts (MoE) architecture with multiple patch tokenizers for efficient and accurate time series forecasting.

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python

.. image:: https://img.shields.io/badge/PyTorch-2.0%2B-red.svg
   :target: https://pytorch.org/
   :alt: PyTorch

.. image:: https://img.shields.io/badge/🤗%20Transformers-4.40.1%2B-yellow.svg
   :target: https://huggingface.co/transformers/
   :alt: Transformers

Quick Start
-----------

Install PatchMoE:

.. code-block:: bash

   pip install patchmoe

Basic usage:

.. code-block:: python

   import torch
   from patchmoe import PatchMoEForPrediction, PatchMoeConfig

   # Create configuration
   config = PatchMoeConfig(
       hidden_size=1024,
       num_attention_heads=16,
       rope_theta=100000,
       tie_word_embeddings=False,
   )

   # Initialize model
   model = PatchMoEForPrediction(config)

   # Generate forecasts
   input_data = torch.randn(32, 2048, 1)  # [batch, sequence, channels]
   forecasts = model.generate(input_data, max_new_tokens=336)
   print(f"Forecast shape: {forecasts.shape}")

Key Features
------------

🎯 **Multi-Scale Patch Processing**
   Utilizes multiple patch sizes for comprehensive temporal pattern capture

⚡ **Mixture of Experts**
   Efficient routing mechanism for scalable model capacity

🔄 **Autoregressive Generation**
   Supports flexible forecast length generation

🧠 **RevIN Normalization**
   Built-in reversible instance normalization for better generalization

🚀 **FlashAttention**
   Optimized attention mechanism for improved efficiency

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/configuration
   user_guide/models
   user_guide/training
   user_guide/inference
   user_guide/evaluation

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_forecasting
   examples/multivariate_forecasting
   examples/fine_tuning
   examples/custom_datasets

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/configuration
   api/models
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/architecture
   advanced/moe_routing
   advanced/patch_tokenization
   advanced/performance_optimization

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   roadmap

Performance
-----------

PatchMoE achieves state-of-the-art performance on multiple time series forecasting benchmarks:

* **Time-Series-Library**: Ranked #1 in MSE/MAE metrics
* **ETT Dataset**: Superior performance across all variants
* **Weather Dataset**: Consistent improvements over baseline models
* **Electricity Dataset**: Excellent scalability for high-dimensional time series

Model Specifications
--------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Specification
     - Value
   * - Architecture
     - Causal Transformer (Decoder-only) with MoE
   * - Parameter Count
     - 2.5B
   * - Number of Layers
     - 12
   * - Hidden Size
     - 1024
   * - Attention Heads
     - 16
   * - Context Length
     - Up to 2880
   * - Patch Sizes
     - [96, 64, 48, 24]
   * - Forecast Heads
     - [24, 96, 336]
   * - Precision
     - FP32/BF16
   * - Optimization
     - FlashAttention, RevIN

Citation
--------

If you find PatchMoE helpful for your research, please cite our paper:

.. code-block:: bibtex

   @article{patchmoe2025,
     title={PatchMoE: A Large-Scale Time Series Foundation Model with Mixture of Experts Architecture},
     author={PatchMoE Team},
     journal={arXiv preprint},
     year={2025}
   }

Support
-------

* **Issues**: `GitHub Issues <https://github.com/patch-moe/patchmoe/issues>`_
* **Discussions**: `GitHub Discussions <https://github.com/patch-moe/patchmoe/discussions>`_
* **Email**: patchmoe@example.com

License
-------

This project is licensed under the Apache License 2.0 - see the `LICENSE <https://github.com/patch-moe/patchmoe/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
