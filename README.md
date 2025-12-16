# Causal Discovery for Predictive Maintenance

Master Project - UFAZ 2025-2026

## Overview

This project applies causal discovery algorithms to industrial predictive maintenance data. We analyze sensor readings from 100 machines over one year to identify causal relationships between sensor measurements, error events, and component failures.

Since there's no ground truth, we use stability-based validation: running algorithms on 100 machines and measuring how often edges appear.

## Dataset

**Source:** Azure Predictive Maintenance Dataset
**Download:** https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance

Put the 5 CSV files in `data/raw/` after downloading.

## Algorithms

10 causal discovery algorithms implemented:

| Algorithm | Type | Library |
|-----------|------|---------|
| GCMVL | Granger Causality | statsmodels |
| VarLiNGAM | Non-Gaussian | lingam |
| PCMCI+ | Constraint-based | tigramite |
| PCGCE | Constraint-based | tigramite |
| Dynotears | Score-based | gcastle |
| TiMINo | Time series | custom |
| NBCB-w, NBCB-e | Hybrid | custom |
| CBNB-w, CBNB-e | Hybrid | custom |

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install gcastle torch
```

## Usage

```bash
# Step 1: Preprocess data
python -m src.preprocessing.data_merger

# Step 2: Run algorithms (~40 min)
python -m src.algorithms.algorithm_runner --all --parallel --workers 8

# Step 3: Calculate stability
python -m src.analysis.stability_calculator

# Step 4: Generate figures
python -m src.analysis.report_visualizations
```

### Docker

```bash
docker compose up --build
```

## Project Structure

```
├── data/
│   ├── raw/              # CSV files (not in git)
│   └── processed/        # Preprocessed data
├── src/
│   ├── preprocessing/    # Data preparation
│   ├── algorithms/       # 10 algorithms
│   ├── analysis/         # Stability analysis
│   └── visualization/    # Plotting
├── results/
│   ├── causal_graphs/    # JSON outputs
│   ├── stability_scores/ # Analysis results
│   └── figures/          # Visualizations
└── requirements.txt
```

## Results

See `results/PROJECT_SUMMARY.md` for details.

- 1,000 runs completed (100 machines x 10 algorithms)
- 499 non-trivial causal edges found
- Error codes predict component failures (up to 88% stability)

## Troubleshooting

**gcastle not found:**
```bash
pip install gcastle torch
```

**Out of memory:**
```bash
python -m src.algorithms.algorithm_runner --all --parallel --workers 4
```

**tigramite import error:**
```bash
pip install tigramite --upgrade
```
