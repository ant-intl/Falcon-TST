"""
Setup script for PatchMoE - A Large-Scale Time Series Foundation Model with Mixture of Experts Architecture
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="patchmoe",
    version="0.1.0",
    author="PatchMoE Team",
    author_email="patchmoe@example.com",
    description="A large-scale time series foundation model utilizing MoE architecture with multiple patch tokenizers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/patch-moe/patchmoe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.0",
        ],
    },
    include_package_data=True,
    package_data={
        "patchmoe": ["figures/*.png"],
    },
    entry_points={
        "console_scripts": [
            "patchmoe-train=patchmoe.trainer.train:main",
            "patchmoe-eval=patchmoe.trainer.evaluate:main",
        ],
    },
)
