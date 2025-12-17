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
python -m src.visualization.plot_results

# Step 5: Generate publication figures (Springer format)
python generate_paper_figures.py
```

### Docker

```bash
docker compose up --build
```

## Project Structure

```
├── data/
│   ├── raw/                    # Original CSV files (download from Kaggle)
│   ├── processed/              # Preprocessed time series per machine
│   └── README.md               # Dataset documentation
├── src/
│   ├── preprocessing/          # Data preparation pipeline
│   ├── algorithms/             # 10 causal discovery algorithms
│   ├── analysis/               # Stability calculation
│   └── visualization/          # Result plotting
├── results/
│   ├── causal_graphs/          # Per-machine JSON outputs (100 files)
│   ├── stability_scores/       # Stability analysis CSVs
│   ├── figures/                # Colorful visualizations (8 figures)
│   └── PROJECT_SUMMARY.md      # Detailed results summary
├── paper_figures/              # Publication-quality figures (Springer format)
├── generate_paper_figures.py   # Grayscale figures for paper
├── generate_extra_figures.py   # Colorful figures for presentations
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Results

See `results/PROJECT_SUMMARY.md` for detailed analysis.

### Key Findings
- **1,000 runs** completed (100 machines × 10 algorithms)
- **30 validated edges** found by 3+ algorithms with ≥70% stability
- **Best algorithm**: Dynotears (71 high-stability edges, 47.3% avg stability)

### Main Discoveries
- Error codes predict component failures (80-88% stability)
- Failures trigger maintenance actions (79-86% stability)
- Causal chain: **Errors → Failures → Maintenance**

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
