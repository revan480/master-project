# Causal Discovery for Predictive Maintenance
## Master's project Results Summary

---

## Executive Summary

This research applies **10 causal discovery algorithms** to the **Azure Predictive Maintenance dataset** to identify causal relationships that can predict machine failures. Using a **stability-based validation approach** across 100 machines, we discovered high-confidence causal edges that reveal actionable insights for predictive maintenance.

### Key Achievements

| Metric | Value |
|--------|-------|
| Total Algorithm Runs | **1,000** (100 machines x 10 algorithms) |
| Machines Analyzed | **100** |
| Variables per Machine | **30** |
| Time Series Length | **366 observations** per machine |
| Total Execution Time | **~22 minutes** (8 parallel workers) |
| Non-trivial Edges Discovered | **499** |
| High-Confidence Edges (>=80% stability) | **72** (Dynotears) |

---

## Algorithm Performance Comparison

| Algorithm | Avg Edges | Avg Time (s) | High Stability Edges | Avg Stability (%) |
|-----------|-----------|--------------|---------------------|-------------------|
| GCMVL | 12.3 | 42.5 | 3 | 12.8 |
| VarLiNGAM | 66.5 | 2.2 | 19 | 8.9 |
| PCMCI+ | 99.1 | 8.8 | 7 | 12.2 |
| PCGCE | 186.2 | 0.5 | 52 | 20.7 |
| **Dynotears** | **112.0** | **21.1** | **72** | **47.6** |
| TiMINo | 435.0 | 1.1 | 0 | 50.0 |
| NBCB-w | 108.2 | 9.5 | 16 | 13.4 |
| NBCB-e | 42.5 | 2.6 | 10 | 9.4 |
| CBNB-w | 108.2 | 9.5 | 16 | 13.4 |
| CBNB-e | 42.1 | 2.6 | 10 | 9.5 |

**Key Finding**: Dynotears (NOTEARS algorithm) achieves the highest number of high-stability edges (72), making it the most reliable for predictive maintenance applications.

---

## Top 10 Predictive Maintenance Edges

These are the most important causal relationships for predicting machine failures:

| Rank | Source | Target | Category | Stability (%) | # Algorithms |
|------|--------|--------|----------|---------------|--------------|
| 1 | **maint_comp2** | **failure_comp2** | Maintenance -> Failure | **89.2** | 4 |
| 2 | **error3_count** | **failure_comp2** | Error -> Failure | **74.4** | 5 |
| 3 | **error1_count** | **failure_comp1** | Error -> Failure | **73.2** | 4 |
| 4 | **error2_count** | **failure_comp2** | Error -> Failure | **72.6** | 5 |
| 5 | **maint_comp1** | **failure_comp1** | Maintenance -> Failure | **82.5** | 4 |
| 6 | **rotate_mean** | **failure_comp2** | Sensor -> Failure | **64.8** | 5 |
| 7 | **volt_mean** | **failure_comp1** | Sensor -> Failure | **63.2** | 4 |
| 8 | rotate_min | failure_comp2 | Sensor -> Failure | 66.0 | 1 |
| 9 | pressure_mean | failure_comp4 | Sensor -> Failure | 65.0 | 1 |
| 10 | error3_count | failure_comp1 | Error -> Failure | 60.0 | 1 |

---

## Key Findings

### 1. Error Codes are Strong Failure Predictors

| Error Type | Predicts Failure | Stability | Algorithms Agreeing |
|------------|-----------------|-----------|---------------------|
| error3_count | Component 2 | 74.4% | 5 |
| error1_count | Component 1 | 73.2% | 4 |
| error2_count | Component 2 | 72.6% | 5 |

**Interpretation**: Error type 3 and error type 2 are strong predictors of Component 2 failures, while error type 1 predicts Component 1 failures. This suggests component-specific error monitoring strategies.

### 2. Sensor Readings Predict Failures

| Sensor | Predicts Failure | Stability | Algorithms Agreeing |
|--------|-----------------|-----------|---------------------|
| rotate_mean | Component 2 | 64.8% | 5 |
| volt_mean | Component 1 | 63.2% | 4 |
| pressure_mean | Component 4 | 65.0% | 1 |

**Interpretation**: Rotation speed anomalies predict Component 2 failures, voltage anomalies predict Component 1 failures, and pressure anomalies predict Component 4 failures.

### 3. Maintenance-Failure Relationship

| Maintenance | Failure | Stability | Interpretation |
|-------------|---------|-----------|----------------|
| maint_comp2 | failure_comp2 | 89.2% | Strong correlation - reactive maintenance |
| maint_comp1 | failure_comp1 | 82.5% | Strong correlation - reactive maintenance |

**Interpretation**: The high stability of maintenance-to-failure edges suggests that maintenance is often performed reactively (after or during failure) rather than proactively.

### 4. Cross-Sensor Relationships

Strong causal relationships exist between different sensor types:
- **vibration_max -> rotate_max**: 98% stability
- **volt_max -> pressure_max**: High cross-sensor influence

These relationships can be used for multi-sensor anomaly detection.

---

## Stability-Based Validation Results

### Edge Stability Distribution

| Stability Range | # Edges | Percentage |
|-----------------|---------|------------|
| High (>=80%) | 72 | 14.4% |
| Medium (50-80%) | 165 | 33.1% |
| Low (<50%) | 262 | 52.5% |

### Algorithm Agreement Analysis

The Jaccard similarity between algorithms ranges from 0.05 to 0.45, indicating:
- Moderate agreement on high-confidence edges
- Each algorithm captures different aspects of causal structure
- Ensemble approach (multiple algorithms) provides more robust results

---

## Methodology Summary

### Data
- **Dataset**: Azure Predictive Maintenance
- **Source**: Microsoft Azure AI Gallery
- **Machines**: 100 industrial machines
- **Time Period**: 1 year of telemetry data
- **Variables**: 30 (telemetry, errors, maintenance, failures)

### Algorithms Implemented

| Category | Algorithms |
|----------|------------|
| Granger-based | GCMVL (VAR + Lasso) |
| Non-Gaussian | VarLiNGAM |
| Constraint-based | PCMCI+, PCGCE |
| Score-based | Dynotears (NOTEARS) |
| Hybrid | TiMINo, NBCB-w, NBCB-e, CBNB-w, CBNB-e |

### Validation Approach

**Stability Score** = (Machines showing edge / Total machines) x 100%

This approach:
- Does not require ground truth
- Validates consistency across independent machines
- Identifies robust causal relationships

---

## Files Structure

```
results/
├── causal_graphs/          # Raw algorithm outputs (100 JSON files)
├── stability_scores/       # Stability analysis results
│   ├── stability_*.csv     # Per-algorithm stability scores
│   ├── high_confidence_edges.csv
│   ├── algorithm_agreement.csv
│   └── top_pdm_edges.csv   # Top predictive maintenance edges
├── figures/                # All generated visualizations
│   ├── fig1_algorithm_comparison.png/pdf
│   ├── fig2_stability_heatmap.png/pdf
│   ├── fig3_algorithm_agreement.png/pdf
│   ├── fig4_consensus_graph.png/pdf
│   ├── fig5_stability_distribution.png/pdf
│   ├── fig6_edge_categories.png/pdf
│   ├── fig7_variable_relationships.png/pdf
│   ├── fig8_execution_boxplot.png/pdf
│   ├── fig9_consensus_by_threshold.png/pdf
│   ├── fig10_summary_table.png/pdf
│   ├── fig_pdm_network.png/pdf
│   └── fig_pdm_category_summary.png/pdf
├── figures_project/         # project-ready figures
├── tables_project/          # LaTeX tables
└── data_project/            # Raw data exports
```

---

## Conclusions

1. **Error codes are the strongest predictors** of machine failures, with up to 74% stability across 100 machines.

2. **Sensor readings (rotation, voltage, pressure)** provide early warning signals for specific component failures.

3. **Dynotears (NOTEARS algorithm)** achieves the best balance of stability and edge discovery.

4. **Stability-based validation** successfully identifies robust causal relationships without requiring ground truth.

5. **Cross-algorithm consensus** (edges found by multiple algorithms) provides the most reliable predictions.

---

## Reproducibility

To reproduce these results:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python -m src.preprocessing.data_preprocessor

# 3. Run all algorithms (parallel)
python -m src.algorithms.algorithm_runner --all --parallel --workers 8

# 4. Calculate stability scores
python -m src.analysis.stability_calculator

# 5. Generate visualizations
python -m src.analysis.project_visualizations

# 6. Extract PdM insights
python -m src.analysis.pdm_insights
```

**Total execution time**: ~25 minutes on AMD Ryzen 7 9700X (8 cores)

---

*Generated: December 2025*
*Master's project - Causal Discovery for Predictive Maintenance*
