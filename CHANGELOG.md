# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and documentation
- Comprehensive test suite with pytest
- GitHub Actions CI/CD pipeline
- Pre-commit hooks for code quality
- Sphinx documentation with RTD theme

### Changed
- Improved code organization and structure
- Enhanced docstrings and type hints
- Optimized model configuration handling

### Fixed
- Various bug fixes and improvements

## [0.1.0] - 2025-01-XX

### Added
- Initial release of PatchMoE
- Core PatchMoE model implementation with MoE architecture
- Multiple patch tokenizers for time series processing
- RevIN (Reversible Instance Normalization) support
- FlashAttention integration for improved efficiency
- Autoregressive generation capabilities
- Comprehensive configuration system
- Basic examples and tutorials
- Unit tests for core functionality
- Documentation and API reference

### Features
- **Model Architecture**: Causal Transformer with Mixture of Experts
- **Patch Processing**: Multi-scale patch tokenization (96, 64, 48, 24)
- **MoE Routing**: Top-k expert routing with configurable parameters
- **Generation**: Flexible forecast length generation
- **Normalization**: Built-in RevIN for better generalization
- **Attention**: FlashAttention for memory efficiency
- **Configuration**: Extensive configuration options
- **Compatibility**: Full HuggingFace Transformers integration

### Model Specifications
- Parameter Count: 2.5B
- Number of Layers: 12
- Hidden Size: 1024
- Attention Heads: 16
- Context Length: Up to 2880
- Forecast Heads: [24, 96, 336]
- Supported Precisions: FP32, BF16, FP16

### Performance
- State-of-the-art results on Time-Series-Library benchmarks
- Superior performance on ETT datasets
- Excellent scalability for high-dimensional time series
- Efficient inference with FlashAttention optimization

### Documentation
- Comprehensive API documentation
- Getting started guide
- Configuration reference
- Example notebooks and scripts
- Performance benchmarking guide

### Development Tools
- Pre-commit hooks for code quality
- GitHub Actions CI/CD pipeline
- Automated testing with pytest
- Code formatting with Black and isort
- Type checking with mypy
- Documentation building with Sphinx

---

## Release Notes Format

Each release includes:

### Added
- New features and capabilities

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Now removed features

### Fixed
- Bug fixes

### Security
- Security improvements

---

## Contributing

When contributing to this project, please:

1. Follow the existing changelog format
2. Add entries under the "Unreleased" section
3. Move entries to a new version section when releasing
4. Use clear, descriptive language
5. Reference issue numbers when applicable

For more details, see our [Contributing Guidelines](CONTRIBUTING.md).
