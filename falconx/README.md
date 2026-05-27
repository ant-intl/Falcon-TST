# Falcon-X: A Multivariate Time-Series Foudation Model

## Installation

Install the package from [PyPI](https://pypi.org/project/falcon-tst/):

```bash
pip install falcon-tst
```

## Usage

### Notation

Throughout this document we follow the conventions below:

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size (number of input series) |
| `L` | Lookback length — number of historical time steps in `context` |
| `H` | Horizon — number of future steps to forecast (`prediction_length`) |
| `C` | Number of channels / variates (multivariate only) |
| `Q` | Number of predicted quantiles (`Q = 21`) |

### Quick Start

```python
import numpy as np
from falcontst import FalconClient

# Prepare inputs
B, L, H = 32, 512, 96
context = np.random.randn(B, L)
input_mask = np.ones_like(context)

client = FalconClient()
result = client.quantile_predict(
    context=context,
    prediction_length=H,
    model_name="Falcon-X",
    input_mask=input_mask,  # 1 = observed, 0 = missing
    is_multivariate=True,
)

print(result)
```

### Output Format

The return value `result` is a dictionary containing the key `prob_prediction`:

```python
prob = np.array(result['prob_prediction'])  # shape: (B, Q, H)
```

Where:
- **B**: batch size (number of input series)
- **Q**: number of quantiles — 21 quantiles: `[0.01, 0.05, 0.10, ..., 0.90, 0.95, 0.99]`
- **H**: prediction horizon (`prediction_length`)

To obtain the point prediction (median), take the middle quantile (index 10):

```python
point_prediction = prob[:, 10, :]  # median (quantile = 0.5)
```

### Multivariate Forecasting

Falcon-X supports multivariate time series via `group_ids` and `is_multivariate`:

| Parameter | Description |
|-----------|-------------|
| `group_ids` | An array of group IDs (one per batch element, must be a **non-decreasing** sequence). Series sharing the same group ID are treated as channels of the same multivariate time series. |
| `is_multivariate` | When set to `True`, `group_ids` is ignored and **all** input series are treated as channels of a single multivariate time series. Defaults to `False`. |

The following example demonstrates how to use `group_ids` to define multivariate groups:

```python
import numpy as np
from falcontst import FalconClient

# Prepare inputs
B, L, H = 8, 512, 96
context = np.random.randn(B, L)
input_mask = np.ones_like(context)

# First 2 series as one multivariate group, next 3 as another, last 3 as another
group_ids = np.array([0, 0, 1, 1, 1, 2, 2, 2])

client = FalconClient()
result = client.quantile_predict(
    context=context,
    prediction_length=H,
    model_name="Falcon-X",
    group_ids=group_ids,
    input_mask=input_mask,  # 1 = observed, 0 = missing
)

print(result)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `context` | `np.ndarray` | Input time series of shape `(B, L)`. Supports `np.nan` for missing values. |
| `prediction_length` | `int` | Forecast horizon `H` (number of future steps to predict). |
| `model_name` | `str \| None` | Model to use. `None` for the default model. |
| `group_ids` | `np.ndarray` | Group IDs of shape `(B,)` for multivariate grouping. Must be non-decreasing. |
| `input_mask` | `np.ndarray` | Binary mask of shape `(B, L)`. `1` = observed, `0` = missing. |
| `is_multivariate` | `bool` | If `True`, treat all series as one multivariate group (ignores `group_ids`). |

### GIFT-Eval Benchmark

Run all cells in [eval/falcon-x.ipynb](eval/falcon-x.ipynb) to reproduce the full GIFT-Eval evaluation. The GIFT-Eval dataset directory must be available — follow the [instructions](https://github.com/SalesforceAIResearch/gift-eval/tree/main#installation) to download it.

Upon completion, two files are generated:

| File | Content |
|------|---------|
| `all_results.csv` | Per-dataset/term MASE and CRPS metrics |
| `agg_results.csv` | Overall geometric-mean aggregation across all datasets |

## Citation
If you find Falcon-X useful in your work, please consider citing:

```bibtex
@article{liu2026falconx,
      title={Falcon-X: A Time Series Foundation Model for Heterogeneous Multivariate Modeling}, 
      author={Yiding Liu and Yifan Hu and Hongjie Xia and Peiyuan Liu and Hongzhou Chen and Xilin Dai and Zewei Dong and Jiang-Ming Yang},
      journal={arXiv preprint arXiv:2605.27286},
      year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
