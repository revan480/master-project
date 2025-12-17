# Data Directory

This directory contains the Azure Predictive Maintenance dataset and processed time series files.

## Dataset Download

Download the dataset from Kaggle:
**URL:** https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance

After downloading, extract and place the CSV files in the `raw/` directory.

## Directory Structure

```
data/
├── raw/                    # Original dataset files (not tracked in git)
│   ├── PdM_telemetry.csv
│   ├── PdM_errors.csv
│   ├── PdM_failures.csv
│   ├── PdM_machines.csv
│   └── PdM_maint.csv
├── processed/              # Generated time series (not tracked in git)
│   ├── machine_001.csv
│   ├── machine_002.csv
│   └── ...
└── README.md              # This file
```

## Raw Data Files

### PdM_telemetry.csv (876,100 rows)
Hourly sensor readings from 100 machines over 1 year.

| Column | Type | Description |
|--------|------|-------------|
| datetime | timestamp | Reading timestamp (hourly) |
| machineID | int | Machine identifier (1-100) |
| volt | float | Voltage reading |
| rotate | float | Rotation speed |
| pressure | float | Pressure reading |
| vibration | float | Vibration measurement |

### PdM_errors.csv (3,919 rows)
Error events logged by machines.

| Column | Type | Description |
|--------|------|-------------|
| datetime | timestamp | Error timestamp |
| machineID | int | Machine identifier |
| errorID | string | Error type (error1-error5) |

### PdM_failures.csv (761 rows)
Component failure events (what we want to predict/explain causally).

| Column | Type | Description |
|--------|------|-------------|
| datetime | timestamp | Failure timestamp |
| machineID | int | Machine identifier |
| failure | string | Failed component (comp1-comp4) |

### PdM_machines.csv (100 rows)
Static machine metadata.

| Column | Type | Description |
|--------|------|-------------|
| machineID | int | Machine identifier |
| model | string | Machine model (model1-model4) |
| age | int | Machine age in years |

### PdM_maint.csv (3,286 rows)
Scheduled and unscheduled maintenance records.

| Column | Type | Description |
|--------|------|-------------|
| datetime | timestamp | Maintenance timestamp |
| machineID | int | Machine identifier |
| comp | string | Component maintained (comp1-comp4) |

## Processed Data Format

After running the preprocessing pipeline, each machine gets its own CSV file with daily aggregated features:

### machine_XXX.csv

| Column | Description |
|--------|-------------|
| date | Date (daily granularity) |
| machineID | Machine identifier |
| volt_mean | Daily mean voltage |
| volt_std | Daily voltage std deviation |
| volt_min | Daily minimum voltage |
| volt_max | Daily maximum voltage |
| rotate_mean | Daily mean rotation |
| rotate_std | Daily rotation std deviation |
| rotate_min | Daily minimum rotation |
| rotate_max | Daily maximum rotation |
| pressure_mean | Daily mean pressure |
| pressure_std | Daily pressure std deviation |
| pressure_min | Daily minimum pressure |
| pressure_max | Daily maximum pressure |
| vibration_mean | Daily mean vibration |
| vibration_std | Daily vibration std deviation |
| vibration_min | Daily minimum vibration |
| vibration_max | Daily maximum vibration |
| error1_count | Count of error1 events |
| error2_count | Count of error2 events |
| error3_count | Count of error3 events |
| error4_count | Count of error4 events |
| error5_count | Count of error5 events |
| failure_comp1 | Binary: comp1 failure occurred |
| failure_comp2 | Binary: comp2 failure occurred |
| failure_comp3 | Binary: comp3 failure occurred |
| failure_comp4 | Binary: comp4 failure occurred |
| maint_comp1 | Binary: comp1 maintenance performed |
| maint_comp2 | Binary: comp2 maintenance performed |
| maint_comp3 | Binary: comp3 maintenance performed |
| maint_comp4 | Binary: comp4 maintenance performed |
| model | Machine model (categorical) |
| age | Machine age |

## Data Statistics

- **Total machines:** 100
- **Time span:** ~1 year (365 days)
- **Expected rows per machine:** ~365
- **Total processed rows:** ~36,500

## Notes

1. Raw data files are NOT tracked in git (too large, ~77 MB)
2. Processed files are NOT tracked in git (regeneratable)
3. Place `.gitkeep` files to preserve directory structure
4. Run preprocessing to generate processed files:
   ```bash
   python -m src.preprocessing.data_merger
   ```

## Quick Start

1. Download dataset from Kaggle link above
2. Extract CSV files to `raw/` folder
3. Run preprocessing command
4. Verify 100 machine files created in `processed/`
