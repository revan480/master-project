# Project Summary: Causal Discovery for Predictive Maintenance

## Overview

This project applies 10 causal discovery algorithms to the Azure Predictive Maintenance dataset to identify causal relationships between sensor readings, error events, component failures, and maintenance actions. Since no ground truth exists, we use **stability-based validation**: running each algorithm on 100 machines and measuring how consistently edges appear across machines.

## Experimental Setup

- **Dataset**: Azure Predictive Maintenance (Kaggle)
- **Machines**: 100 industrial machines
- **Time span**: ~1 year (365 days per machine)
- **Variables**: 30 time series features per machine
  - 16 sensor statistics (volt, rotate, pressure, vibration - mean/std/min/max)
  - 5 error counts (error1-error5)
  - 4 failure indicators (comp1-comp4)
  - 4 maintenance indicators (comp1-comp4)
  - 1 machine age

## Algorithms Evaluated

| Algorithm | Type | Library | Edges Found | High-Stability (>=80%) | Avg Stability |
|-----------|------|---------|-------------|------------------------|---------------|
| GCMVL | Granger Causality | statsmodels | 96 | 3 | 12.8% |
| VarLiNGAM | Non-Gaussian | lingam | 749 | 19 | 8.8% |
| PCMCI+ | Constraint-based | tigramite | 813 | 7 | 12.2% |
| PCGCE | Constraint-based | tigramite | 899 | 52 | 20.7% |
| **Dynotears** | Score-based | gcastle | 237 | **71** | **47.3%** |
| TiMINo | Time series | custom | 870 | 0 | 50.0% |
| NBCB-w | Hybrid | custom | 809 | 16 | 13.4% |
| NBCB-e | Hybrid | custom | 443 | 10 | 9.6% |
| CBNB-w | Hybrid | custom | 809 | 16 | 13.4% |
| CBNB-e | Hybrid | custom | 429 | 10 | 9.9% |

**Best performer**: Dynotears with 71 high-stability edges and 47.3% average stability.

## Key Findings

### Validated Causal Relationships (>=80% stability or >=70% with 3+ algorithms)

1. **Error -> Failure relationships**:
   - error3_count -> failure_comp2 (88% stability, 3 algorithms)
   - error2_count -> failure_comp2 (85% stability, 3 algorithms)
   - error1_count -> failure_comp1 (80% stability, 3 algorithms)

2. **Failure -> Maintenance relationships**:
   - failure_comp2 -> maint_comp2 (86% stability, 3 algorithms)
   - failure_comp1 -> maint_comp1 (79% stability, 3 algorithms)

3. **Sensor autocorrelations** (expected, validates methodology):
   - pressure_mean -> pressure_mean (99% stability, 5 algorithms)
   - rotate_mean -> rotate_mean (99% stability, 5 algorithms)
   - Sensor min/max/std relationships consistently discovered

### Domain Validation

The discovered causal chain **Errors -> Failures -> Maintenance** aligns with industrial domain knowledge:
- Error events precede component failures (predictive signal)
- Failures trigger maintenance actions (reactive maintenance pattern)

## Run Statistics

- **Total algorithm runs**: 1,000 (100 machines x 10 algorithms)
- **Total unique edges discovered**: 499 (non-trivial)
- **High-stability edges (>=80%)**: 204
- **Cross-algorithm validated edges**: 30 (found by 3+ algorithms)

## Stability Thresholds

| Stability Level | Range | Interpretation |
|-----------------|-------|----------------|
| High | >=80% | Confident causal relationship |
| Moderate | 70-79% | Likely real, needs multi-algorithm support |
| Low | <70% | Possibly spurious or machine-specific |

## Files Generated

```
results/
├── causal_graphs/              # Per-machine JSON outputs (100 files)
│   └── machine_XXX.json        # Edges discovered per algorithm
├── stability_scores/           # Stability analysis
│   ├── stability_*.csv         # Per-algorithm stability scores (10 files)
│   ├── high_confidence_edges.csv
│   ├── algorithm_agreement.csv # Jaccard similarity matrix
│   └── summary_report.json     # Complete statistics
└── PROJECT_SUMMARY.md          # This file
```

## Reproducibility

```bash
# Full pipeline
python -m src.preprocessing.data_merger              # Step 1: Preprocess data
python -m src.algorithms.algorithm_runner --all --parallel --workers 8  # Step 2: Run algorithms
python -m src.analysis.stability_calculator          # Step 3: Calculate stability
python -m src.visualization.plot_results             # Step 4: Generate figures
```
