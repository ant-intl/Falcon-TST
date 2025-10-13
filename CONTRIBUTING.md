# Contributing to PatchMoE

We welcome contributions to PatchMoE! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0
- Git

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/patchmoe.git
   cd patchmoe
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

6. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Types of Contributions

We welcome several types of contributions:

### 🐛 Bug Reports

When filing a bug report, please include:
- A clear description of the issue
- Steps to reproduce the problem
- Expected vs. actual behavior
- Environment details (Python version, PyTorch version, etc.)
- Minimal code example that demonstrates the issue

### ✨ Feature Requests

For feature requests, please provide:
- A clear description of the proposed feature
- Use case and motivation
- Possible implementation approach (if you have ideas)

### 🔧 Code Contributions

We accept contributions for:
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements

## 🛠️ Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks with:
```bash
# Format code
black patchmoe tests examples
isort patchmoe tests examples

# Check linting
flake8 patchmoe tests examples

# Type checking
mypy patchmoe
```

### Testing

We use pytest for testing. Please ensure:

1. **Write tests** for new functionality
2. **Run existing tests** to ensure nothing breaks:
   ```bash
   pytest tests/
   ```

3. **Maintain test coverage**:
   ```bash
   pytest --cov=patchmoe tests/
   ```

### Documentation

- Add docstrings to all public functions and classes
- Follow Google-style docstrings
- Update README.md if adding new features
- Add examples for new functionality

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for custom patch sizes
fix: resolve memory leak in attention mechanism
docs: update installation instructions
test: add unit tests for MoE routing
```

Prefixes:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `style:` - Code style changes

## 🧪 Testing Guidelines

### Unit Tests

Write unit tests for:
- Individual functions and methods
- Model components
- Configuration validation
- Utility functions

### Integration Tests

Write integration tests for:
- End-to-end model functionality
- Model loading and saving
- Generation pipeline

### Test Structure

```python
import pytest
import torch
from patchmoe import PatchMoeConfig, PatchMoEForPrediction


class TestPatchMoEModel:
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        config = PatchMoeConfig()
        model = PatchMoEForPrediction(config)
        assert model is not None
    
    def test_forward_pass(self):
        """Test model forward pass."""
        config = PatchMoeConfig(hidden_size=64, seq_length=128)
        model = PatchMoEForPrediction(config)
        
        batch_size, seq_len = 2, 128
        input_ids = torch.randn(batch_size, seq_len)
        
        with torch.no_grad():
            output = model(input_ids, max_output_length=24)
        
        assert output.shape == (batch_size, 24)
```

## 📋 Pull Request Process

1. **Ensure your code passes all checks**:
   ```bash
   # Run pre-commit hooks
   pre-commit run --all-files
   
   # Run tests
   pytest tests/
   ```

2. **Update documentation** if needed

3. **Create a pull request** with:
   - Clear title and description
   - Reference to related issues
   - List of changes made
   - Screenshots/examples if applicable

4. **Respond to review feedback** promptly

5. **Ensure CI passes** before requesting final review

## 🏷️ Release Process

Releases are handled by maintainers:

1. Version bumping follows [Semantic Versioning](https://semver.org/)
2. Changelog is updated with notable changes
3. GitHub releases are created with release notes
4. PyPI packages are published automatically

## 📚 Documentation

### Building Documentation

```bash
cd docs/
pip install -r requirements.txt
make html
```

### Documentation Structure

- API documentation is auto-generated from docstrings
- Tutorials and guides are written in Markdown
- Examples are provided as executable Python scripts

## 🤝 Community Guidelines

### Code of Conduct

We follow a code of conduct to ensure a welcoming environment:

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Report inappropriate behavior

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private matters (patchmoe@example.com)

## 🎯 Areas for Contribution

We particularly welcome contributions in:

### High Priority
- Performance optimizations
- Memory efficiency improvements
- Additional time series datasets support
- Better error handling and validation

### Medium Priority
- Documentation improvements
- Example notebooks and tutorials
- Integration with other libraries
- Visualization tools

### Low Priority
- Code style improvements
- Minor bug fixes
- Test coverage improvements

## 📊 Benchmarking

When contributing performance improvements:

1. **Benchmark before and after** your changes
2. **Use consistent hardware** for comparisons
3. **Include benchmark results** in your PR
4. **Test on multiple datasets** if possible

Example benchmarking script:
```python
import time
import torch
from patchmoe import PatchMoEForPrediction, PatchMoeConfig

def benchmark_model(config, input_shape, num_runs=10):
    model = PatchMoEForPrediction(config)
    model.eval()
    
    input_data = torch.randn(*input_shape)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(input_data, max_new_tokens=96)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model.generate(input_data, max_new_tokens=96)
            times.append(time.time() - start)
    
    return {
        'mean_time': sum(times) / len(times),
        'std_time': torch.std(torch.tensor(times)).item()
    }
```

## 🔍 Review Process

### For Contributors

- Be patient during the review process
- Address feedback constructively
- Ask questions if feedback is unclear
- Update your PR based on suggestions

### For Reviewers

- Provide constructive, specific feedback
- Explain the reasoning behind suggestions
- Acknowledge good practices
- Be respectful and encouraging

## 📈 Metrics and Goals

We track several metrics to ensure project health:

- **Code Coverage**: Aim for >90%
- **Documentation Coverage**: All public APIs documented
- **Performance**: No regressions in benchmarks
- **Compatibility**: Support for specified Python/PyTorch versions

## 🙏 Recognition

Contributors are recognized through:

- GitHub contributor graphs
- Release notes acknowledgments
- Hall of fame in documentation
- Special recognition for significant contributions

Thank you for contributing to PatchMoE! 🎉
