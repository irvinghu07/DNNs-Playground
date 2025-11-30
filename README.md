# COMP5511 Assignment 2

This repository contains implementations for various machine learning tasks including regression, denoising autoencoders, residual CNNs, and reinforcement learning.

## Project Structure

```
├── Dataset/
│   ├── ParisHousing.csv
│   └── Fashion-MNIST/
│       ├── train_clean.csv
│       ├── train_noisy.csv
│       ├── test_clean.csv
│       └── test_noisy.csv
├── task1.ipynb          # Linear Regression on Paris Housing
├── task2.ipynb          # Denoising Autoencoder
├── task3.ipynb          # Residual Denoising CNN
├── task4.ipynb          # Q-Learning on FrozenLake
├── gifs/                # Generated GIF animations
├── plots/               # Generated plots
├── qtables/             # Saved Q-tables
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project configuration
└── REPORT_TASK*.md      # Task reports
```

## Dependencies

### Core ML/DL Framework
- **PyTorch 2.4.0** - Deep learning framework used for Tasks 2 & 3

### Data Science & Numerical Computing
- **numpy** (2.3.4) - Numerical computing
- **pandas** (2.3.3) - Data manipulation and analysis
- **scipy** (1.16.3) - Scientific computing

### Machine Learning
- **scikit-learn** (1.7.2) - Classical ML algorithms (Task 1)
- **torch** (PyTorch 2.4.0) - Deep learning (Tasks 2 & 3)

### Reinforcement Learning
- **gymnasium** (>=1.2.2) - RL environment framework (Task 4)
- **pygame** (>=2.6.1) - Required for Gymnasium rendering

### Visualization
- **matplotlib** (>=3.10.7) - Plotting and visualization
- **imageio** (>=2.37.2) - Creating GIF animations

### Development Tools
- **jupyter/ipykernel** (7.1.0) - Jupyter notebook support
- **tqdm** (>=4.67.1) - Progress bars
- **ipython** (9.7.0) - Enhanced Python REPL

### Optional Dependencies
- **pickle5** - Enhanced pickle support (fallback to standard pickle if unavailable)

## Installation

### Using pip (from requirements.txt)
```bash
pip install -r requirements.txt
```

### Using uv (from pyproject.toml)
```bash
uv pip install -e .
```

### PyTorch Installation
For GPU support, install PyTorch 2.4.0 with CUDA:
```bash
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only:
```bash
pip install torch==2.4.0 torchvision torchaudio
```

## Tasks Overview

### Task 1: Linear Regression (task1.ipynb)
- Dataset: Paris Housing prices
- Libraries: pandas, scikit-learn, numpy, matplotlib
- Implements baseline and feature-selected linear regression models

### Task 2: Denoising Autoencoder (task2.ipynb)
- Dataset: Fashion-MNIST (noisy/clean pairs)
- Libraries: PyTorch 2.4.0, pandas, numpy, matplotlib
- CNN-based autoencoder architecture with encoder-decoder structure

### Task 3: Residual Denoising CNN (task3.ipynb)
- Dataset: Fashion-MNIST (noisy/clean pairs)
- Libraries: PyTorch 2.4.0, pandas, numpy, matplotlib
- Residual learning approach for direct noise prediction

### Task 4: Q-Learning on FrozenLake (task4.ipynb)
- Environment: Gymnasium FrozenLake-v1 (8×8 grid)
- Libraries: gymnasium, numpy, pandas, matplotlib, imageio, tqdm
- Implements Q-learning for both deterministic and stochastic environments

## Running the Notebooks

### Local Environment
1. Ensure all dependencies are installed
2. Update `DATASET_ROOT` paths in notebooks if needed:
   - Tasks 2 & 3: Change from `/kaggle/input/denoising` to your local path
   - Task 1: Verify `Dataset/ParisHousing.csv` path

### Kaggle Environment
Notebooks are pre-configured for Kaggle with:
- `DATASET_ROOT = "/kaggle/input/denoising"` (Tasks 2 & 3)
- GPU support automatically detected

### Jupyter Notebook
```bash
jupyter notebook
```

## Hardware Requirements

### Minimum
- CPU: Any modern multi-core processor
- RAM: 8GB
- Storage: 2GB free space

### Recommended (for Tasks 2 & 3)
- GPU: NVIDIA GPU with CUDA support
- VRAM: 4GB+
- RAM: 16GB+

## Python Version

Requires **Python ≥ 3.11**

## Notes

- Tasks 2 & 3 automatically detect and use CUDA if available
- Task 4 uses CPU-based Q-learning (no GPU required)
- GIF recordings and plots are saved to respective directories
- Q-tables can be saved/loaded for Task 4 experiments

## License

Academic project for COMP5511 course.
