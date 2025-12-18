# Causal Discovery for Predictive Maintenance

Master Project - UFAZ 2025-2026

## Overview

This project applies causal discovery algorithms to industrial predictive maintenance data. We analyze sensor readings from 100 machines over one year to identify causal relationships between sensor measurements, error events, and component failures.

Since there's no ground truth for real industrial data, we use **stability-based validation**: running algorithms on 100 machines and measuring how consistently edges appear across machines.

## Dataset

**Source:** Azure Predictive Maintenance Dataset (Microsoft)

**Download Options:**
- Kaggle: https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance

**Files Required (place in `data/raw/`):**
| File | Description | Size |
|------|-------------|------|
| `PdM_telemetry.csv` | Hourly sensor readings (volt, rotate, pressure, vibration) | 876,100 rows |
| `PdM_errors.csv` | Error events (error1-error5) | 3,919 rows |
| `PdM_failures.csv` | Component failures (comp1-comp4) | 761 rows |
| `PdM_machines.csv` | Machine metadata (model, age) | 100 rows |
| `PdM_maint.csv` | Maintenance records | 3,286 rows |

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
# Clone the repository
git clone https://github.com/your-username/MasterProject_PdM_Causal_Discovery.git
cd MasterProject_PdM_Causal_Discovery

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install gcastle torch
```

## Usage

### Step-by-Step

```bash
# Step 1: Download data and place in data/raw/

# Step 2: Preprocess data (creates 100 machine files)
python -m src.preprocessing.data_merger

# Step 3: Run algorithms (~40 min on 8 cores)
python -m src.algorithms.algorithm_runner --all --parallel --workers 8

# Step 4: Calculate stability scores
python -m src.analysis.stability_calculator

# Step 5: Generate visualizations
python -m src.visualization.plot_results

# Step 6: Generate dataset figures (optional)
python generate_dataset_figures.py

# Step 7: Generate paper figures (optional)
python generate_paper_figures.py
python generate_extra_figures.py
```

### Docker

```bash
# Run full pipeline
docker compose up --build

# Run specific service
docker compose up preprocess
docker compose up algorithms
docker compose up analysis
docker compose up visualize
```

## Project Structure

```
MasterProject_PdM_Causal_Discovery/
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
│   ├── figures/                # Generated visualizations
│   └── PROJECT_SUMMARY.md      # Detailed results summary
├── paper_figures/              # Publication-quality figures
├── generate_dataset_figures.py # Dataset visualization script
├── generate_paper_figures.py   # Paper figure generation
├── generate_extra_figures.py   # Presentation figures
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Results

See `results/PROJECT_SUMMARY.md` for detailed analysis.

### Key Findings
- **1,000 runs** completed (100 machines x 10 algorithms)
- **30 validated edges** found by 3+ algorithms with >= 70% stability
- **Best algorithm**: Dynotears (71 high-stability edges, 47.3% avg stability)

### Main Discoveries
- Error codes predict component failures (80-88% stability)
- Failures trigger maintenance actions (79-86% stability)
- Causal chain: **Errors -> Failures -> Maintenance**

### Generated Figures

| Category | Location | Description |
|----------|----------|-------------|
| Algorithm Results | `results/figures/` | Stability distributions, agreement heatmaps, consensus graphs |
| Dataset Overview | `results/figures/dataset/` | Data characteristics, correlations, event analysis |
| Paper Figures | `paper_figures/` | Publication-quality figures for Springer |

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