# Causal Discovery for Predictive Maintenance

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-Academic-green.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)

**Master Project - Final Deliverable**

**Authors:** Ravan Osmanli & Anar Abdullazada  
**University:** UFAZ (Universite Franco-Azerbaidjanaise)  
**Program:** Master 2 - Data Science and Artificial Intelligence  
**Academic Year:** 2025-2026  
**Course:** Master Project  

---

## Project Overview

This project applies causal discovery algorithms to industrial predictive maintenance data. We analyze sensor readings from 100 machines over one year to identify causal relationships between sensor measurements, error events, and component failures.

The main challenge is the absence of ground truth causal graphs. We address this through stability analysis: running algorithms independently on 100 machines and measuring consensus across results.

### What We Did

1. Preprocessed the Azure PdM dataset (5 CSV files, 876k+ rows)
2. Implemented 10 causal discovery algorithms
3. Ran each algorithm on each of the 100 machines
4. Calculated stability scores to validate findings

### Main Findings

- Error codes are strong predictors of failures (up to 88% stability)
- Rotation and voltage sensors predict specific component failures
- Multiple algorithms agree on key causal relationships

## Dataset

- **Source:** Azure Predictive Maintenance Dataset (Kaggle)
- **Scale:** 100 machines, 366 days, hourly measurements
- **Sensors:** Voltage, rotation, pressure, vibration
- **Events:** Error logs, failure records, maintenance activities

Download from: https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance

## Algorithms

We implemented 10 causal discovery algorithms:

| Algorithm | Type | Library |
|-----------|------|---------|
| GCMVL | Granger Causality | statsmodels |
| VarLiNGAM | Non-Gaussian | lingam |
| PCMCI+ | Constraint-based | tigramite |
| PCGCE | Constraint-based | tigramite |
| Dynotears | Score-based (NOTEARS) | gcastle |
| TiMINo | Time series | custom |
| NBCB-w, NBCB-e | Hybrid | custom |
| CBNB-w, CBNB-e | Hybrid | custom |

Each algorithm runs independently on each machine. Edges appearing consistently across machines are considered reliable.

## Installation

### Requirements

- Python 3.9 or higher
- 8GB RAM minimum
- About 500MB disk space

### Setup

```bash
# Clone repository
git clone https://github.com/[username]/masterproject-pdm-causal.git
cd masterproject-pdm-causal

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Using Docker

```bash
docker-compose up --build
```

## Usage

### Step 1: Prepare the data

Put the Kaggle CSV files in `data/raw/`, then run:

```bash
python -m src.preprocessing.data_preprocessor
```

This creates one file per machine in `data/processed/`.

### Step 2: Run algorithms

```bash
# Run all algorithms on all machines (takes ~40 minutes)
python -m src.algorithms.algorithm_runner --all --parallel --workers 8

# Or run on a single machine for testing
python -m src.algorithms.algorithm_runner --machine 1
```

### Step 3: Calculate stability scores

```bash
python -m src.analysis.stability_calculator
```

### Step 4: Generate figures

```bash
python -m src.analysis.thesis_visualizations
```

## Project Structure

```
MasterProject_PdM_Causal_Discovery/
├── data/
│   ├── raw/              # Original CSV files (not in git)
│   └── processed/        # Preprocessed time series
├── src/
│   ├── preprocessing/    # Data merging scripts
│   ├── algorithms/       # 10 causal discovery algorithms
│   ├── analysis/         # Stability calculation
│   └── visualization/    # Figure generation
├── results/
│   ├── causal_graphs/    # Algorithm outputs (JSON)
│   ├── stability_scores/ # Analysis results
│   ├── figures/          # Visualizations
│   └── figures_report/   # Report-ready figures
├── notebooks/            # Jupyter notebooks
└── requirements.txt
```

## Results

See `results/PROJECT_SUMMARY.md` for detailed findings.

Key results:
- 1,000 algorithm runs completed (100 machines x 10 algorithms)
- 499 non-trivial causal edges discovered
- Top predictor: error3_count -> failure_comp2 (74.4% stability)

## Troubleshooting

**gcastle not found:**
```bash
pip install gcastle torch
```

**Out of memory:**
Use fewer parallel workers:
```bash
python -m src.algorithms.algorithm_runner --all --parallel --workers 4
```

**tigramite import error:**
```bash
pip install tigramite --upgrade
```

## References

1. Runge, J., et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. Science Advances.
2. Shimizu, S., et al. (2006). A linear non-Gaussian acyclic model for causal discovery. JMLR.
3. Pamfil, R., et al. (2020). DYNOTEARS: Structure Learning from Time-Series Data. AISTATS.

## How to Cite

If you use this code, please cite:

```
Osmanli, R., & Abdullazada, A. (2025). Causal Discovery for Predictive
Maintenance: A Stability-Based Validation Approach. Master Project,
UFAZ, Baku, Azerbaijan.
```

## License

Academic project - UFAZ Master Project 2025-2026. For educational purposes only.

## Acknowledgments

- Azure AI for the Predictive Maintenance dataset
- tigramite, lingam, and gcastle library developers
- Our supervisor for guidance throughout the project
