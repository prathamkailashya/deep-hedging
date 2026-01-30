#!/usr/bin/env python3
"""
Setup script for Deep Hedging Research Pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="deep-hedging",
    version="1.0.0",
    author="Research Pipeline",
    description="Deep Hedging Research Pipeline - Replication and Extensions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
        "finance": [
            "yfinance>=0.1.70",
        ],
        "optuna": [
            "optuna>=2.10.0",
        ],
        "signature": [
            "signatory>=1.2.0",
        ],
        "rl": [
            "stable-baselines3>=1.5.0",
            "gymnasium>=0.26.0",
        ],
        "full": [
            "yfinance>=0.1.70",
            "optuna>=2.10.0",
            "mlflow>=1.20.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    keywords="deep-learning hedging options finance neural-networks",
)
