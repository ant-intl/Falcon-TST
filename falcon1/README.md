<div align="center">

# Falcon-1.0: A Large-Scale Time Series Foundation Model

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40.1-yellow.svg)](https://huggingface.co/transformers/)

</div>

## 📖 Introduction

Falcon-1.0 is a cutting-edge time series foundation model that leverages the power of Mixture of Experts (MoE) architecture combined with multiple patch tokenizers. This innovative approach enables efficient processing of time series data while maintaining high accuracy across various forecasting tasks.

### Key Features

- **Multi-Scale Patch Tokenizers**: Utilizes multiple patch tokenizers to capture comprehensive temporal patterns
- **Mixture of Experts**: Efficient routing mechanism for scalable model capacity
- **Flexible Input / Output Length**: Supports arbitrary input and output length


<div align="center">
<img src="figures/falcon_tst_framework.png" alt="description" width="100%">
</div>

<!-- ![Falcon-TST Architecture Overview](figures/falcon_tst_framework.png) -->

### Model Configurations

We have currently developed three different-sized Falcon-TST models, as detailed in the table below:

<div align="center">
<img src="figures/model_configuration.png" alt="description" width="100%">
</div>

### Performance

Falcon-TST achieves state-of-the-art zero-shot results on well-acknowledged long-term forecasting benchmarks.

<div align="center">
<img src="figures/zero_shot.png" alt="description" width="100%">
</div>


### Inference Time on different devices

The following figure domonstrates the inference time of Falcon-TST on different devices. As a billion-scale foundation model, Falcon-TST<sub>Large</sub> can make forecasts within 1 second even on CPU devices.

<div align="center">
<img src="figures/inference_time.png" alt="description" width="70%">
</div>


## 🚀 Quick Start

### Installation

Install the following dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- **transformers == 4.40.1**

### Code Example

```python
import torch
from transformers import AutoModel

# Load pre-trained model (when available)
model = AutoModel.from_pretrained(
    'ant-intl/Falcon-TST_Large',
    trust_remote_code=True
)

# Prepare your time series data
batch_size, lookback_length, channels = 1, 2880, 7
time_series = torch.randn(batch_size, lookback_length, channels)

# Load the model and data to the same device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
model = model.to(device)
time_series = time_series.to(device)

# Generate forecasts
forecast_length = 96
predictions = model.predict(time_series, forecast_horizon=forecast_length)

print(predictions)
```

### Evaluation

+ Prepare the benchmark datasets

You can access the well-acknowledged long-term forecast datasets from [[Google Drive]](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy), and place the datasets under `./dataset`.

+ Benchmark evaluation on ETTh1

```bash
python run_eval.py --ckpt_path ant-intl/Falcon-TST_Large --root_path ./dataset --test_data_list ETTh1
```

### Advanced Usage

More advanced usages like Megatron-LM-based pre-training and fine-tuning procedure, and flexible data distributed sampling of our datasets are coming soon.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
