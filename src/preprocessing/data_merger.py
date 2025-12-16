"""
Data Merger Module

Converts raw PdM CSV files into daily aggregated time series per machine.

Usage:
    # Process all machines (full dataset)
    python -m src.preprocessing.data_merger

    # Test mode: 3 machines, 30 days
    python -m src.preprocessing.data_merger --test

    # Process specific machines
    python -m src.preprocessing.data_merger --machines 1,2,3,4,5
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from tqdm import tqdm
import click


class DataMerger:
    """
    Merges and preprocesses Azure Predictive Maintenance data.

    Combines telemetry, errors, failures, maintenance, and machine metadata
    into daily aggregated time series for each machine.
    """

    def __init__(self, raw_data_path: str, processed_data_path: str,
                 test_mode: bool = False):
        """
        Initialize the DataMerger.

        Args:
            raw_data_path: Path to directory containing raw CSV files
            processed_data_path: Path to directory for output files
            test_mode: If True, only process first 30 days of data
        """
        self.raw_path = Path(raw_data_path)
        self.processed_path = Path(processed_data_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.test_mode = test_mode

        # Test mode settings
        self.test_start_date = pd.Timestamp('2015-01-01')
        self.test_end_date = pd.Timestamp('2015-01-30')

        # Data containers
        self.telemetry: Optional[pd.DataFrame] = None
        self.errors: Optional[pd.DataFrame] = None
        self.failures: Optional[pd.DataFrame] = None
        self.maintenance: Optional[pd.DataFrame] = None
        self.machines: Optional[pd.DataFrame] = None

    def load_raw_data(self, machine_ids: Optional[List[int]] = None) -> None:
        """
        Load all raw CSV files into memory.

        Args:
            machine_ids: Optional list of machine IDs to filter (for efficiency)
        """
        print("=" * 60)
        print("LOADING RAW DATA")
        print("=" * 60)

        if self.test_mode:
            print(f"TEST MODE: Loading data from {self.test_start_date.date()} to {self.test_end_date.date()}")
            if machine_ids:
                print(f"TEST MODE: Filtering for machines: {machine_ids}")

        # Load telemetry
        telemetry_path = self.raw_path / "PdM_telemetry.csv"
        if telemetry_path.exists():
            print(f"\nLoading telemetry from: {telemetry_path}")
            self.telemetry = pd.read_csv(telemetry_path, parse_dates=['datetime'])

            # Filter for test mode
            if self.test_mode:
                self.telemetry = self.telemetry[
                    (self.telemetry['datetime'] >= self.test_start_date) &
                    (self.telemetry['datetime'] <= self.test_end_date)
                ]
            if machine_ids:
                self.telemetry = self.telemetry[self.telemetry['machineID'].isin(machine_ids)]

            print(f"  Telemetry: {len(self.telemetry):,} rows")
        else:
            raise FileNotFoundError(f"Telemetry file not found: {telemetry_path}")

        # Load errors
        errors_path = self.raw_path / "PdM_errors.csv"
        if errors_path.exists():
            self.errors = pd.read_csv(errors_path, parse_dates=['datetime'])
            if self.test_mode:
                self.errors = self.errors[
                    (self.errors['datetime'] >= self.test_start_date) &
                    (self.errors['datetime'] <= self.test_end_date)
                ]
            if machine_ids:
                self.errors = self.errors[self.errors['machineID'].isin(machine_ids)]
            print(f"  Errors: {len(self.errors):,} rows")

        # Load failures
        failures_path = self.raw_path / "PdM_failures.csv"
        if failures_path.exists():
            self.failures = pd.read_csv(failures_path, parse_dates=['datetime'])
            if self.test_mode:
                self.failures = self.failures[
                    (self.failures['datetime'] >= self.test_start_date) &
                    (self.failures['datetime'] <= self.test_end_date)
                ]
            if machine_ids:
                self.failures = self.failures[self.failures['machineID'].isin(machine_ids)]
            print(f"  Failures: {len(self.failures):,} rows")

        # Load maintenance
        maint_path = self.raw_path / "PdM_maint.csv"
        if maint_path.exists():
            self.maintenance = pd.read_csv(maint_path, parse_dates=['datetime'])
            if self.test_mode:
                self.maintenance = self.maintenance[
                    (self.maintenance['datetime'] >= self.test_start_date) &
                    (self.maintenance['datetime'] <= self.test_end_date)
                ]
            if machine_ids:
                self.maintenance = self.maintenance[self.maintenance['machineID'].isin(machine_ids)]
            print(f"  Maintenance: {len(self.maintenance):,} rows")

        # Load machines
        machines_path = self.raw_path / "PdM_machines.csv"
        if machines_path.exists():
            self.machines = pd.read_csv(machines_path)
            if machine_ids:
                self.machines = self.machines[self.machines['machineID'].isin(machine_ids)]
            print(f"  Machines: {len(self.machines):,} rows")

        print("=" * 60)

    def aggregate_telemetry(self, machine_id: int) -> pd.DataFrame:
        """
        Aggregate telemetry data to daily statistics for a single machine.

        Args:
            machine_id: Machine identifier

        Returns:
            DataFrame with daily aggregated sensor statistics
        """
        # Filter for this machine
        machine_telemetry = self.telemetry[self.telemetry['machineID'] == machine_id].copy()
        machine_telemetry['date'] = machine_telemetry['datetime'].dt.date

        # Aggregate by date
        agg_funcs = {
            'volt': ['mean', 'std', 'min', 'max'],
            'rotate': ['mean', 'std', 'min', 'max'],
            'pressure': ['mean', 'std', 'min', 'max'],
            'vibration': ['mean', 'std', 'min', 'max']
        }

        daily = machine_telemetry.groupby('date').agg(agg_funcs)

        # Flatten column names
        daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
        daily = daily.reset_index()
        daily['machineID'] = machine_id

        return daily

    def add_error_counts(self, daily_df: pd.DataFrame, machine_id: int) -> pd.DataFrame:
        """Add daily error counts for each error type."""
        if self.errors is None or len(self.errors) == 0:
            # Add empty error columns
            for i in range(1, 6):
                daily_df[f'error{i}_count'] = 0
            return daily_df

        machine_errors = self.errors[self.errors['machineID'] == machine_id].copy()

        if len(machine_errors) == 0:
            for i in range(1, 6):
                daily_df[f'error{i}_count'] = 0
            return daily_df

        machine_errors['date'] = machine_errors['datetime'].dt.date

        # Count errors by type and date
        error_counts = machine_errors.groupby(['date', 'errorID']).size().unstack(fill_value=0)
        error_counts.columns = [f'{col}_count' for col in error_counts.columns]
        error_counts = error_counts.reset_index()

        # Merge with daily data
        daily_df = daily_df.merge(error_counts, on='date', how='left')

        # Ensure all error columns exist and fill missing with 0
        for i in range(1, 6):
            col_name = f'error{i}_count'
            if col_name not in daily_df.columns:
                daily_df[col_name] = 0

        error_cols = [col for col in daily_df.columns if col.endswith('_count')]
        daily_df[error_cols] = daily_df[error_cols].fillna(0).astype(int)

        return daily_df

    def add_failure_indicators(self, daily_df: pd.DataFrame, machine_id: int) -> pd.DataFrame:
        """Add binary failure indicators for each component."""
        # Initialize all failure columns to 0
        for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
            daily_df[f'failure_{comp}'] = 0

        if self.failures is None or len(self.failures) == 0:
            return daily_df

        machine_failures = self.failures[self.failures['machineID'] == machine_id].copy()

        if len(machine_failures) == 0:
            return daily_df

        machine_failures['date'] = machine_failures['datetime'].dt.date

        # Create binary indicators for each failure type
        for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
            comp_failures = machine_failures[machine_failures['failure'] == comp]['date'].unique()
            daily_df[f'failure_{comp}'] = daily_df['date'].isin(comp_failures).astype(int)

        return daily_df

    def add_maintenance_indicators(self, daily_df: pd.DataFrame, machine_id: int) -> pd.DataFrame:
        """Add binary maintenance indicators for each component."""
        # Initialize all maintenance columns to 0
        for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
            daily_df[f'maint_{comp}'] = 0

        if self.maintenance is None or len(self.maintenance) == 0:
            return daily_df

        machine_maint = self.maintenance[self.maintenance['machineID'] == machine_id].copy()

        if len(machine_maint) == 0:
            return daily_df

        machine_maint['date'] = machine_maint['datetime'].dt.date

        # Create binary indicators for each maintenance type
        for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
            comp_maint = machine_maint[machine_maint['comp'] == comp]['date'].unique()
            daily_df[f'maint_{comp}'] = daily_df['date'].isin(comp_maint).astype(int)

        return daily_df

    def add_machine_metadata(self, daily_df: pd.DataFrame, machine_id: int) -> pd.DataFrame:
        """Add static machine metadata."""
        if self.machines is None or len(self.machines) == 0:
            daily_df['model'] = 'unknown'
            daily_df['age'] = 0
            return daily_df

        machine_info = self.machines[self.machines['machineID'] == machine_id]
        if len(machine_info) == 0:
            daily_df['model'] = 'unknown'
            daily_df['age'] = 0
        else:
            machine_info = machine_info.iloc[0]
            daily_df['model'] = machine_info['model']
            daily_df['age'] = machine_info['age']

        return daily_df

    def process_machine(self, machine_id: int) -> pd.DataFrame:
        """
        Process all data for a single machine.

        Args:
            machine_id: Machine identifier

        Returns:
            Complete daily time series DataFrame for the machine
            (all numeric columns, no nulls, ready for causal discovery)
        """
        # Start with telemetry aggregation
        daily = self.aggregate_telemetry(machine_id)

        # Add error counts
        daily = self.add_error_counts(daily, machine_id)

        # Add failure indicators
        daily = self.add_failure_indicators(daily, machine_id)

        # Add maintenance indicators
        daily = self.add_maintenance_indicators(daily, machine_id)

        # Add machine metadata
        daily = self.add_machine_metadata(daily, machine_id)

        # Sort by date
        daily = daily.sort_values('date').reset_index(drop=True)

        # =====================================================================
        # CLEANUP: Prepare data for causal discovery algorithms
        # =====================================================================

        # Drop non-numeric columns (date, machineID, model)
        # Keep 'age' as it's numeric and potentially useful
        columns_to_drop = ['date', 'machineID', 'model']
        daily = daily.drop(columns=[c for c in columns_to_drop if c in daily.columns])

        # Fill null values with 0 (std of single value = NaN -> 0)
        daily = daily.fillna(0)

        return daily

    def process_all_machines(self, machine_ids: Optional[List[int]] = None,
                             file_prefix: str = "machine") -> Dict[int, pd.DataFrame]:
        """
        Process all machines and save to individual CSV files.

        Args:
            machine_ids: Optional list of machine IDs to process.
                        If None, processes all machines (1-100).
            file_prefix: Prefix for output files (default: "machine")

        Returns:
            Dictionary mapping machine_id to processed DataFrame
        """
        if machine_ids is None:
            machine_ids = list(range(1, 101))

        results = {}

        print(f"\nProcessing {len(machine_ids)} machines...")
        for machine_id in tqdm(machine_ids, desc="Processing machines"):
            try:
                daily_df = self.process_machine(machine_id)
                results[machine_id] = daily_df

                # Save to CSV
                output_path = self.processed_path / f"{file_prefix}_{machine_id:03d}.csv"
                daily_df.to_csv(output_path, index=False)

            except Exception as e:
                print(f"\nError processing machine {machine_id}: {e}")

        print(f"\nProcessed {len(results)} machines successfully.")
        print(f"Output saved to: {self.processed_path}")

        return results

    def create_combined_dataset(self, machine_ids: Optional[List[int]] = None,
                                file_prefix: str = "machine") -> pd.DataFrame:
        """
        Create a single combined dataset with all machines.

        Args:
            machine_ids: Optional list of machine IDs to include
            file_prefix: Prefix for input files

        Returns:
            Combined DataFrame with all machines
        """
        if machine_ids is None:
            machine_ids = list(range(1, 101))

        all_data = []

        for machine_id in tqdm(machine_ids, desc="Combining machines"):
            file_path = self.processed_path / f"{file_prefix}_{machine_id:03d}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)

        # Save combined dataset
        output_path = self.processed_path / f"all_{file_prefix}s_combined.csv"
        combined.to_csv(output_path, index=False)
        print(f"Combined dataset saved to: {output_path}")

        return combined

    def print_summary(self, results: Dict[int, pd.DataFrame]) -> None:
        """Print summary statistics for processed data."""
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)

        if not results:
            print("No data processed.")
            return

        # Get a sample DataFrame to show structure
        sample_id = list(results.keys())[0]
        sample_df = results[sample_id]

        print(f"\nMachines processed: {len(results)}")
        print(f"Machine IDs: {sorted(results.keys())}")

        print(f"\n--- Data Structure (Machine {sample_id}) ---")
        print(f"Rows (days): {len(sample_df)}")
        print(f"Columns: {len(sample_df.columns)}")

        print(f"\n--- Column Names ---")
        for i, col in enumerate(sample_df.columns):
            dtype = sample_df[col].dtype
            nulls = sample_df[col].isnull().sum()
            print(f"  {i+1:2d}. {col:<25} ({dtype}, nulls: {nulls})")

        print(f"\n--- Null Counts Across All Machines ---")
        total_nulls = 0
        for machine_id, df in results.items():
            nulls = df.isnull().sum().sum()
            total_nulls += nulls
            if nulls > 0:
                print(f"  Machine {machine_id}: {nulls} null values")

        if total_nulls == 0:
            print("  No null values found!")
        else:
            print(f"  Total null values: {total_nulls}")

        print(f"\n--- Sample Data (Machine {sample_id}, first 5 rows) ---")
        print(sample_df.head().to_string())

        print(f"\n--- Numeric Columns Summary ---")
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric features for causal discovery: {len(numeric_cols)}")
        print(f"  {numeric_cols}")

        # =====================================================================
        # VALIDATION CHECKS: Ensure data is ready for causal discovery
        # =====================================================================
        print(f"\n--- Validation Checks (Machine {sample_id}) ---")

        # Check 1: All columns are numeric
        all_numeric = sample_df.select_dtypes(include=[np.number]).shape[1] == sample_df.shape[1]
        non_numeric_cols = sample_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if all_numeric:
            print(f"  [OK] All columns are numeric: True")
        else:
            print(f"  [FAIL] All columns are numeric: False")
            print(f"    Non-numeric columns: {non_numeric_cols}")

        # Check 2: Zero null values
        null_count = sample_df.isnull().sum().sum()
        if null_count == 0:
            print(f"  [OK] Zero null values: True")
        else:
            print(f"  [FAIL] Zero null values: False ({null_count} nulls found)")

        # Check 3: Final shape
        print(f"  [OK] Final shape: {sample_df.shape}")

        # Overall validation status
        print(f"\n--- Overall Validation ---")
        if all_numeric and null_count == 0:
            print("  [OK] Data is READY for causal discovery algorithms!")
        else:
            print("  [FAIL] Data has issues - please fix before running algorithms")

        print("\n" + "=" * 60)


@click.command()
@click.option('--raw-path', default='data/raw', help='Path to raw data directory')
@click.option('--processed-path', default='data/processed', help='Path to processed data directory')
@click.option('--machines', default=None, help='Comma-separated list of machine IDs (default: all)')
@click.option('--combine', is_flag=True, help='Also create combined dataset')
@click.option('--test', 'test_mode', is_flag=True,
              help='Test mode: process only machines 1,2,3 for first 30 days')
def main(raw_path: str, processed_path: str, machines: Optional[str],
         combine: bool, test_mode: bool):
    """
    Preprocess PdM data: merge and aggregate to daily time series.

    Examples:
        # Full processing (all 100 machines, all data)
        python -m src.preprocessing.data_merger

        # Test mode (3 machines, 30 days)
        python -m src.preprocessing.data_merger --test

        # Specific machines
        python -m src.preprocessing.data_merger --machines 1,2,3,4,5

        # Full processing with combined output
        python -m src.preprocessing.data_merger --combine
    """
    # Handle test mode
    if test_mode:
        print("\n" + "=" * 60)
        print("RUNNING IN TEST MODE")
        print("=" * 60)
        print("- Processing machines: 1, 2, 3")
        print("- Date range: 2015-01-01 to 2015-01-30")
        print("- Output files: test_machine_001.csv, etc.")
        print("=" * 60)

        machine_ids = [1, 2, 3]
        file_prefix = "test_machine"
    else:
        # Parse machine IDs if provided
        machine_ids = None
        if machines:
            machine_ids = [int(m.strip()) for m in machines.split(',')]
        file_prefix = "machine"

    # Initialize merger
    merger = DataMerger(raw_path, processed_path, test_mode=test_mode)

    # Load raw data
    merger.load_raw_data(machine_ids=machine_ids)

    # Process machines
    results = merger.process_all_machines(machine_ids, file_prefix=file_prefix)

    # Print summary
    merger.print_summary(results)

    # Optionally create combined dataset
    if combine:
        merger.create_combined_dataset(machine_ids, file_prefix=file_prefix)


if __name__ == '__main__':
    main()
