"""
Generate Dataset Visualization Figures
Shows how different variables in the PdM dataset relate to each other
Covers: sensor readings, errors, failures, maintenance, and their relationships

For Master's Thesis: Causal Discovery for Predictive Maintenance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory
output_dir = Path('results/figures/dataset')
output_dir.mkdir(exist_ok=True, parents=True)

print("=" * 70)
print("GENERATING DATASET VISUALIZATION FIGURES")
print("Showing how variables affect each other in PdM data")
print("=" * 70)


# =============================================================================
# LOAD DATA
# =============================================================================
def load_all_data():
    """Load raw and processed data."""
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')

    # Load raw data files
    print("\nLoading raw data...")
    telemetry = pd.read_csv(raw_dir / 'PdM_telemetry.csv')
    errors = pd.read_csv(raw_dir / 'PdM_errors.csv')
    failures = pd.read_csv(raw_dir / 'PdM_failures.csv')
    machines = pd.read_csv(raw_dir / 'PdM_machines.csv')
    maint = pd.read_csv(raw_dir / 'PdM_maint.csv')

    # Parse dates
    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
    errors['datetime'] = pd.to_datetime(errors['datetime'])
    failures['datetime'] = pd.to_datetime(failures['datetime'])
    maint['datetime'] = pd.to_datetime(maint['datetime'])

    print(f"  Telemetry: {len(telemetry):,} rows")
    print(f"  Errors: {len(errors):,} rows")
    print(f"  Failures: {len(failures):,} rows")
    print(f"  Machines: {len(machines)} machines")
    print(f"  Maintenance: {len(maint):,} rows")

    # Load processed data (sample of machines)
    print("\nLoading processed data...")
    processed_data = []
    for i in [1, 25, 50, 75, 100]:
        file_path = processed_dir / f'machine_{i:03d}.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['machine_id'] = i
            processed_data.append(df)

    if processed_data:
        processed_df = pd.concat(processed_data, ignore_index=True)
        print(f"  Processed sample: {len(processed_df)} rows from 5 machines")
    else:
        processed_df = None

    return telemetry, errors, failures, machines, maint, processed_df


# =============================================================================
# FIGURE 1: Dataset Overview - Data Structure
# =============================================================================
def create_dataset_overview(telemetry, errors, failures, machines, maint):
    """Create visual overview of the dataset structure."""
    print("\n[1/6] Generating Dataset Overview...")

    fig = plt.figure(figsize=(16, 10))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # === Panel A: Data Sources Diagram ===
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3)

    # Draw data source boxes
    sources = [
        ('Telemetry', '876,100 rows\n4 sensors × hourly', '#3498db', 1),
        ('Errors', '3,919 events\n5 error types', '#e67e22', 3),
        ('Failures', '761 events\n4 components', '#e74c3c', 5),
        ('Maintenance', '3,286 events\n4 components', '#27ae60', 7),
        ('Machines', '100 machines\nAge + Model', '#9b59b6', 9),
    ]

    for name, desc, color, x in sources:
        rect = plt.Rectangle((x-0.8, 0.5), 1.6, 2, facecolor=color,
                              edgecolor='white', linewidth=2, alpha=0.85)
        ax1.add_patch(rect)
        ax1.text(x, 2, name, ha='center', va='center', fontsize=11,
                fontweight='bold', color='white')
        ax1.text(x, 1.2, desc, ha='center', va='center', fontsize=8, color='white')

    ax1.set_title('Azure Predictive Maintenance Dataset Structure',
                  fontsize=14, fontweight='bold', pad=10)

    # === Panel B: Timeline ===
    ax2 = fig.add_subplot(gs[0, 2])

    # Timeline data
    timeline_data = {
        'Start Date': '2015-01-01',
        'End Date': '2016-01-01',
        'Duration': '365 days',
        'Granularity': 'Hourly (raw)',
        'Processed': 'Daily aggregated'
    }

    ax2.axis('off')
    y_pos = 0.9
    ax2.text(0.5, 1.0, 'Timeline Info', ha='center', fontsize=12,
             fontweight='bold', transform=ax2.transAxes)
    for key, val in timeline_data.items():
        ax2.text(0.1, y_pos, f'{key}:', ha='left', fontsize=10,
                transform=ax2.transAxes, fontweight='bold')
        ax2.text(0.55, y_pos, val, ha='left', fontsize=10,
                transform=ax2.transAxes)
        y_pos -= 0.15

    # === Panel C: Sensor Distributions ===
    ax3 = fig.add_subplot(gs[1, 0])

    sensor_means = telemetry.groupby('machineID')[['volt', 'rotate', 'pressure', 'vibration']].mean()

    # Normalize for comparison
    sensor_norm = (sensor_means - sensor_means.min()) / (sensor_means.max() - sensor_means.min())

    bp = ax3.boxplot([sensor_norm['volt'], sensor_norm['rotate'],
                      sensor_norm['pressure'], sensor_norm['vibration']],
                     labels=['Voltage', 'Rotation', 'Pressure', 'Vibration'],
                     patch_artist=True)

    colors = ['#3498db', '#e67e22', '#27ae60', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_ylabel('Normalized Value', fontsize=10)
    ax3.set_title('Sensor Reading Distributions\n(Normalized, per machine)', fontsize=11, fontweight='bold')

    # === Panel D: Error Distribution ===
    ax4 = fig.add_subplot(gs[1, 1])

    error_counts = errors['errorID'].value_counts().sort_index()
    bars = ax4.bar(range(len(error_counts)), error_counts.values,
                   color='#e67e22', edgecolor='white', linewidth=1.5)
    ax4.set_xticks(range(len(error_counts)))
    ax4.set_xticklabels(error_counts.index, fontsize=9)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('Error Event Distribution\n(Total across all machines)', fontsize=11, fontweight='bold')

    for bar, val in zip(bars, error_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(val), ha='center', fontsize=9, fontweight='bold')

    # === Panel E: Failure Distribution ===
    ax5 = fig.add_subplot(gs[1, 2])

    failure_counts = failures['failure'].value_counts().sort_index()
    bars = ax5.bar(range(len(failure_counts)), failure_counts.values,
                   color='#e74c3c', edgecolor='white', linewidth=1.5)
    ax5.set_xticks(range(len(failure_counts)))
    ax5.set_xticklabels(['Comp 1', 'Comp 2', 'Comp 3', 'Comp 4'], fontsize=9)
    ax5.set_ylabel('Count', fontsize=10)
    ax5.set_title('Component Failure Distribution\n(Total across all machines)', fontsize=11, fontweight='bold')

    for bar, val in zip(bars, failure_counts.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', fontsize=9, fontweight='bold')

    # === Panel F: Machine Age Distribution ===
    ax6 = fig.add_subplot(gs[2, 0])

    age_counts = machines['age'].value_counts().sort_index()
    ax6.bar(age_counts.index, age_counts.values, color='#9b59b6',
            edgecolor='white', linewidth=1, alpha=0.8)
    ax6.set_xlabel('Machine Age (years)', fontsize=10)
    ax6.set_ylabel('Number of Machines', fontsize=10)
    ax6.set_title('Machine Age Distribution', fontsize=11, fontweight='bold')

    # === Panel G: Model Distribution ===
    ax7 = fig.add_subplot(gs[2, 1])

    model_counts = machines['model'].value_counts().sort_index()
    colors = ['#3498db', '#e67e22', '#27ae60', '#e74c3c']
    wedges, texts, autotexts = ax7.pie(model_counts.values, labels=model_counts.index,
                                        autopct='%1.0f%%', colors=colors,
                                        explode=[0.02]*len(model_counts))
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    ax7.set_title('Machine Model Distribution', fontsize=11, fontweight='bold')

    # === Panel H: Events Timeline ===
    ax8 = fig.add_subplot(gs[2, 2])

    # Group events by month
    errors['month'] = errors['datetime'].dt.to_period('M')
    failures['month'] = failures['datetime'].dt.to_period('M')
    maint['month'] = maint['datetime'].dt.to_period('M')

    error_monthly = errors.groupby('month').size()
    failure_monthly = failures.groupby('month').size()
    maint_monthly = maint.groupby('month').size()

    months = range(12)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax8.plot(months, error_monthly.values[:12], 'o-', label='Errors',
             color='#e67e22', linewidth=2, markersize=6)
    ax8.plot(months, failure_monthly.values[:12], 's-', label='Failures',
             color='#e74c3c', linewidth=2, markersize=6)
    ax8.plot(months, maint_monthly.values[:12], '^-', label='Maintenance',
             color='#27ae60', linewidth=2, markersize=6)

    ax8.set_xticks(months)
    ax8.set_xticklabels(month_labels, fontsize=8)
    ax8.set_xlabel('Month (2015)', fontsize=10)
    ax8.set_ylabel('Event Count', fontsize=10)
    ax8.set_title('Monthly Event Timeline', fontsize=11, fontweight='bold')
    ax8.legend(loc='upper right', fontsize=8)

    plt.suptitle('Azure PdM Dataset Overview', fontsize=16, fontweight='bold', y=1.02)

    plt.savefig(output_dir / 'dataset_overview.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: dataset_overview.png")


# =============================================================================
# FIGURE 2: Correlation Matrix - How Variables Relate
# =============================================================================
def create_correlation_matrix(processed_df):
    """Create correlation heatmap showing relationships between variables."""
    print("\n[2/6] Generating Correlation Matrix...")

    if processed_df is None:
        print("  Skipped: No processed data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Select columns for correlation
    sensor_cols = ['volt_mean', 'rotate_mean', 'pressure_mean', 'vibration_mean']
    error_cols = ['error1_count', 'error2_count', 'error3_count', 'error4_count', 'error5_count']
    failure_cols = ['failure_comp1', 'failure_comp2', 'failure_comp3', 'failure_comp4']
    maint_cols = ['maint_comp1', 'maint_comp2', 'maint_comp3', 'maint_comp4']

    # === Panel A: Full Correlation Matrix ===
    ax1 = axes[0]

    all_cols = sensor_cols + error_cols + failure_cols + maint_cols + ['age']
    corr_matrix = processed_df[all_cols].corr()

    # Rename for readability
    rename_dict = {
        'volt_mean': 'Voltage', 'rotate_mean': 'Rotation',
        'pressure_mean': 'Pressure', 'vibration_mean': 'Vibration',
        'error1_count': 'Error1', 'error2_count': 'Error2',
        'error3_count': 'Error3', 'error4_count': 'Error4', 'error5_count': 'Error5',
        'failure_comp1': 'Fail1', 'failure_comp2': 'Fail2',
        'failure_comp3': 'Fail3', 'failure_comp4': 'Fail4',
        'maint_comp1': 'Maint1', 'maint_comp2': 'Maint2',
        'maint_comp3': 'Maint3', 'maint_comp4': 'Maint4',
        'age': 'Age'
    }
    corr_matrix.rename(index=rename_dict, columns=rename_dict, inplace=True)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 7}, cbar_kws={'shrink': 0.8},
                ax=ax1)

    ax1.set_title('Full Variable Correlation Matrix\n(Pearson Correlation)',
                  fontsize=12, fontweight='bold')

    # === Panel B: Key Relationships Highlighted ===
    ax2 = axes[1]

    # Focus on error-failure-maintenance relationships
    key_cols = error_cols + failure_cols + maint_cols
    key_corr = processed_df[key_cols].corr()

    key_rename = {
        'error1_count': 'Error 1', 'error2_count': 'Error 2',
        'error3_count': 'Error 3', 'error4_count': 'Error 4', 'error5_count': 'Error 5',
        'failure_comp1': 'Failure 1', 'failure_comp2': 'Failure 2',
        'failure_comp3': 'Failure 3', 'failure_comp4': 'Failure 4',
        'maint_comp1': 'Maint 1', 'maint_comp2': 'Maint 2',
        'maint_comp3': 'Maint 3', 'maint_comp4': 'Maint 4',
    }
    key_corr.rename(index=key_rename, columns=key_rename, inplace=True)

    sns.heatmap(key_corr, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, vmin=-0.5, vmax=0.5,
                linewidths=1, linecolor='white',
                annot_kws={'size': 9, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8}, ax=ax2)

    ax2.set_title('Error-Failure-Maintenance Correlations\n(Key Relationships for PdM)',
                  fontsize=12, fontweight='bold')

    # Add dividing lines to separate categories
    ax2.axhline(y=5, color='black', linewidth=2)
    ax2.axhline(y=9, color='black', linewidth=2)
    ax2.axvline(x=5, color='black', linewidth=2)
    ax2.axvline(x=9, color='black', linewidth=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: correlation_matrix.png")


# =============================================================================
# FIGURE 3: Time Series Patterns
# =============================================================================
def create_time_series_patterns(telemetry):
    """Show temporal patterns in sensor data."""
    print("\n[3/6] Generating Time Series Patterns...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Sample one machine for detailed view
    machine_data = telemetry[telemetry['machineID'] == 1].copy()
    machine_data = machine_data.sort_values('datetime')

    sensors = ['volt', 'rotate', 'pressure', 'vibration']
    titles = ['Voltage (V)', 'Rotation Speed (RPM)', 'Pressure (PSI)', 'Vibration (mm/s)']
    colors = ['#3498db', '#e67e22', '#27ae60', '#9b59b6']

    for idx, (sensor, title, color) in enumerate(zip(sensors, titles, colors)):
        ax = axes[idx // 2, idx % 2]

        # Plot daily mean
        daily = machine_data.set_index('datetime')[sensor].resample('D').mean()

        ax.plot(daily.index, daily.values, color=color, linewidth=0.8, alpha=0.7)

        # Add rolling average
        rolling = daily.rolling(window=7).mean()
        ax.plot(rolling.index, rolling.values, color='red', linewidth=2,
                label='7-day moving avg')

        # Add mean line
        ax.axhline(y=daily.mean(), color='gray', linestyle='--',
                   linewidth=1.5, label=f'Mean: {daily.mean():.1f}')

        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} Over Time (Machine 1)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Sensor Time Series Patterns (One Year)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_dir / 'time_series_patterns.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: time_series_patterns.png")


# =============================================================================
# FIGURE 4: Event Analysis - Errors Leading to Failures
# =============================================================================
def create_event_analysis(errors, failures, maint):
    """Analyze relationship between errors, failures, and maintenance."""
    print("\n[4/6] Generating Event Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # === Panel A: Error Types Before Failures ===
    ax1 = axes[0, 0]

    # For each failure, count errors in the preceding week
    error_before_failure = {f'error{i}': [] for i in range(1, 6)}

    for _, fail_row in failures.iterrows():
        machine_id = fail_row['machineID']
        fail_time = fail_row['datetime']

        # Get errors for this machine in the week before failure
        week_before = fail_time - pd.Timedelta(days=7)
        machine_errors = errors[(errors['machineID'] == machine_id) &
                                (errors['datetime'] >= week_before) &
                                (errors['datetime'] < fail_time)]

        for i in range(1, 6):
            count = len(machine_errors[machine_errors['errorID'] == f'error{i}'])
            error_before_failure[f'error{i}'].append(count)

    # Calculate average errors before failure
    avg_errors = {k: np.mean(v) if v else 0 for k, v in error_before_failure.items()}

    bars = ax1.bar(range(5), list(avg_errors.values()),
                   color=['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#9b59b6'],
                   edgecolor='white', linewidth=2)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['Error 1', 'Error 2', 'Error 3', 'Error 4', 'Error 5'])
    ax1.set_ylabel('Average Count', fontsize=11)
    ax1.set_title('Average Errors in Week Before Failure\n(Predictive Signal Analysis)',
                  fontsize=12, fontweight='bold')

    for bar, val in zip(bars, avg_errors.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

    # === Panel B: Failure to Maintenance Time ===
    ax2 = axes[0, 1]

    # Calculate time from failure to next maintenance for same component
    time_to_maint = []

    for _, fail_row in failures.head(200).iterrows():  # Sample for speed
        machine_id = fail_row['machineID']
        fail_time = fail_row['datetime']
        comp = fail_row['failure']

        # Find next maintenance for same component
        machine_maint = maint[(maint['machineID'] == machine_id) &
                              (maint['comp'] == comp) &
                              (maint['datetime'] > fail_time)]

        if len(machine_maint) > 0:
            next_maint = machine_maint['datetime'].min()
            days_diff = (next_maint - fail_time).days
            if days_diff <= 30:  # Only consider reasonable timeframes
                time_to_maint.append(days_diff)

    if time_to_maint:
        ax2.hist(time_to_maint, bins=15, color='#27ae60', edgecolor='white',
                 linewidth=1.5, alpha=0.8)
        ax2.axvline(x=np.mean(time_to_maint), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(time_to_maint):.1f} days')
        ax2.set_xlabel('Days', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Time from Failure to Maintenance\n(Response Time Analysis)',
                      fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')

    # === Panel C: Failure Distribution by Component ===
    ax3 = axes[1, 0]

    # Failures per machine by component
    failure_by_comp = failures.groupby(['machineID', 'failure']).size().unstack(fill_value=0)

    failure_by_comp.plot(kind='box', ax=ax3,
                         color=dict(boxes='#e74c3c', whiskers='gray',
                                   medians='black', caps='gray'),
                         patch_artist=True)
    ax3.set_xlabel('Component', fontsize=11)
    ax3.set_ylabel('Failures per Machine', fontsize=11)
    ax3.set_title('Failure Rate Distribution by Component\n(Which Components Fail Most?)',
                  fontsize=12, fontweight='bold')

    # === Panel D: Event Chain Visualization ===
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 6)

    # Draw causal chain
    chain_elements = [
        ('Sensor\nAnomaly', 1.5, 4.5, '#3498db'),
        ('Error\nLogged', 4, 4.5, '#e67e22'),
        ('Component\nFailure', 6.5, 4.5, '#e74c3c'),
        ('Maintenance\nAction', 9, 4.5, '#27ae60'),
    ]

    for name, x, y, color in chain_elements:
        circle = plt.Circle((x, y), 0.8, color=color, ec='white', linewidth=3)
        ax4.add_patch(circle)
        ax4.text(x, y, name, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')

    # Draw arrows
    for i in range(len(chain_elements) - 1):
        x1 = chain_elements[i][1] + 0.9
        x2 = chain_elements[i+1][1] - 0.9
        ax4.annotate('', xy=(x2, 4.5), xytext=(x1, 4.5),
                    arrowprops=dict(arrowstyle='-|>', color='gray', lw=3))

    # Add statistics
    ax4.text(5, 2.5, 'PdM Event Chain', ha='center', fontsize=14,
             fontweight='bold', color='#2c3e50')

    stats_text = f"""
Total Errors: {len(errors):,}
Total Failures: {len(failures):,}
Total Maintenance: {len(maint):,}
Avg Time to Maint: {np.mean(time_to_maint):.1f} days
    """
    ax4.text(5, 1.3, stats_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                      edgecolor='#bdc3c7', linewidth=1))

    ax4.set_title('Predictive Maintenance Event Chain', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'event_analysis.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: event_analysis.png")


# =============================================================================
# FIGURE 5: Sensor Behavior Analysis
# =============================================================================
def create_sensor_analysis(telemetry, failures):
    """Analyze how sensor readings relate to failures."""
    print("\n[5/6] Generating Sensor Behavior Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # === Panel A: Sensor Distributions ===
    ax1 = axes[0, 0]

    sensors = ['volt', 'rotate', 'pressure', 'vibration']
    data_for_violin = [telemetry[s].values[::100] for s in sensors]  # Sample for speed

    parts = ax1.violinplot(data_for_violin, positions=range(4), showmeans=True, showmedians=True)

    colors = ['#3498db', '#e67e22', '#27ae60', '#9b59b6']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax1.set_xticks(range(4))
    ax1.set_xticklabels(['Voltage\n(V)', 'Rotation\n(RPM)', 'Pressure\n(PSI)', 'Vibration\n(mm/s)'])
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Sensor Reading Distributions', fontsize=12, fontweight='bold')

    # === Panel B: Sensor Values Before vs After Failure ===
    ax2 = axes[0, 1]

    # Compare sensor readings 24h before failure vs normal operation
    before_failure_data = []
    normal_data = []

    for _, fail_row in failures.head(100).iterrows():  # Sample
        machine_id = fail_row['machineID']
        fail_time = fail_row['datetime']

        # Get data 24h before failure
        before = fail_time - pd.Timedelta(hours=24)
        before_data = telemetry[(telemetry['machineID'] == machine_id) &
                                (telemetry['datetime'] >= before) &
                                (telemetry['datetime'] < fail_time)]

        if len(before_data) > 0:
            before_failure_data.append(before_data[['volt', 'rotate', 'pressure', 'vibration']].mean())

    # Get normal operation data (random sample)
    normal_sample = telemetry.sample(1000)[['volt', 'rotate', 'pressure', 'vibration']]

    if before_failure_data:
        before_df = pd.DataFrame(before_failure_data)

        x = np.arange(4)
        width = 0.35

        normal_means = normal_sample.mean()
        before_means = before_df.mean()

        # Normalize for comparison
        normal_norm = normal_means / normal_means
        before_norm = before_means / normal_means

        bars1 = ax2.bar(x - width/2, normal_norm, width, label='Normal Operation',
                        color='#27ae60', alpha=0.8)
        bars2 = ax2.bar(x + width/2, before_norm, width, label='24h Before Failure',
                        color='#e74c3c', alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(['Voltage', 'Rotation', 'Pressure', 'Vibration'])
        ax2.set_ylabel('Normalized Value (Normal = 1.0)', fontsize=11)
        ax2.set_title('Sensor Readings: Normal vs Pre-Failure\n(Anomaly Detection)',
                      fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5)

    # === Panel C: Sensor Correlation Scatter ===
    ax3 = axes[1, 0]

    sample = telemetry.sample(5000)
    scatter = ax3.scatter(sample['volt'], sample['vibration'],
                          c=sample['pressure'], cmap='RdYlGn',
                          alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax3, label='Pressure')
    ax3.set_xlabel('Voltage (V)', fontsize=11)
    ax3.set_ylabel('Vibration (mm/s)', fontsize=11)
    ax3.set_title('Sensor Relationships\n(Color = Pressure)', fontsize=12, fontweight='bold')

    # === Panel D: Machine Variability ===
    ax4 = axes[1, 1]

    # Calculate coefficient of variation per machine
    machine_cv = telemetry.groupby('machineID')[sensors].std() / telemetry.groupby('machineID')[sensors].mean()
    machine_cv = machine_cv.mean(axis=1).sort_values(ascending=False)

    top_20 = machine_cv.head(20)
    bottom_20 = machine_cv.tail(20)

    x = np.arange(20)
    width = 0.35

    ax4.bar(x - width/2, top_20.values, width, label='Most Variable', color='#e74c3c', alpha=0.8)
    ax4.bar(x + width/2, bottom_20.values, width, label='Most Stable', color='#27ae60', alpha=0.8)

    ax4.set_xlabel('Machine Rank', fontsize=11)
    ax4.set_ylabel('Coefficient of Variation', fontsize=11)
    ax4.set_title('Machine Sensor Variability\n(Top 20 Variable vs Top 20 Stable)',
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'sensor_analysis.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: sensor_analysis.png")


# =============================================================================
# FIGURE 6: Variable Interaction Summary
# =============================================================================
def create_interaction_summary(processed_df, errors, failures, maint):
    """Create summary of how all variables interact."""
    print("\n[6/6] Generating Variable Interaction Summary...")

    fig = plt.figure(figsize=(18, 12))

    # Main title
    fig.suptitle('How Variables Affect Each Other in Predictive Maintenance',
                 fontsize=16, fontweight='bold', y=0.98)

    # Create custom layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    # === Panel A: Causal Flow Diagram ===
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 4)

    # Draw the causal flow
    layers = [
        ('SENSORS', ['Voltage', 'Rotation', 'Pressure', 'Vibration'], '#3498db', 2),
        ('ERRORS', ['Error 1-5'], '#e67e22', 6),
        ('FAILURES', ['Comp 1-4'], '#e74c3c', 10),
        ('MAINTENANCE', ['Scheduled\nRepairs'], '#27ae60', 14),
    ]

    for layer_name, items, color, x in layers:
        # Layer box
        rect = plt.Rectangle((x-1.5, 0.5), 3, 3, facecolor=color,
                              edgecolor='white', linewidth=2, alpha=0.15)
        ax1.add_patch(rect)

        # Layer title
        ax1.text(x, 3.3, layer_name, ha='center', fontsize=12,
                fontweight='bold', color=color)

        # Items
        for i, item in enumerate(items):
            y = 2.5 - i * 0.5
            ax1.text(x, y, item, ha='center', fontsize=10, color='#2c3e50')

    # Draw arrows between layers
    arrow_style = dict(arrowstyle='-|>', color='#7f8c8d', lw=2,
                       connectionstyle='arc3,rad=0')

    for i in range(len(layers) - 1):
        x1 = layers[i][3] + 1.5
        x2 = layers[i+1][3] - 1.5
        ax1.annotate('', xy=(x2, 2), xytext=(x1, 2), arrowprops=arrow_style)

        # Add relationship label
        mid_x = (x1 + x2) / 2
        if i == 0:
            label = 'Anomaly\nDetection'
        elif i == 1:
            label = 'Predictive\nSignal'
        else:
            label = 'Response\nAction'
        ax1.text(mid_x, 2.6, label, ha='center', fontsize=9,
                style='italic', color='#7f8c8d')

    ax1.set_title('Causal Flow in Predictive Maintenance', fontsize=13,
                  fontweight='bold', pad=10)

    # === Panel B: Error-Failure Relationship ===
    ax2 = fig.add_subplot(gs[1, 0])

    # Create error-failure co-occurrence matrix
    error_failure_matrix = np.zeros((5, 4))

    for _, fail_row in failures.iterrows():
        machine_id = fail_row['machineID']
        fail_time = fail_row['datetime']
        fail_comp = int(fail_row['failure'].replace('comp', '')) - 1

        # Count errors in week before
        week_before = fail_time - pd.Timedelta(days=7)
        machine_errors = errors[(errors['machineID'] == machine_id) &
                                (errors['datetime'] >= week_before) &
                                (errors['datetime'] < fail_time)]

        for _, err_row in machine_errors.iterrows():
            err_type = int(err_row['errorID'].replace('error', '')) - 1
            error_failure_matrix[err_type, fail_comp] += 1

    # Normalize
    error_failure_matrix = error_failure_matrix / error_failure_matrix.sum(axis=0, keepdims=True)
    error_failure_matrix = np.nan_to_num(error_failure_matrix)

    sns.heatmap(error_failure_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=['Comp 1', 'Comp 2', 'Comp 3', 'Comp 4'],
                yticklabels=['Error 1', 'Error 2', 'Error 3', 'Error 4', 'Error 5'],
                ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Error Types Before Each Failure\n(Normalized Frequency)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Failed Component')
    ax2.set_ylabel('Error Type')

    # === Panel C: Failure-Maintenance Relationship ===
    ax3 = fig.add_subplot(gs[1, 1])

    # Calculate maintenance frequency after each failure type
    fail_maint_counts = np.zeros((4, 4))

    for _, fail_row in failures.iterrows():
        machine_id = fail_row['machineID']
        fail_time = fail_row['datetime']
        fail_comp = int(fail_row['failure'].replace('comp', '')) - 1

        # Find maintenance in next 30 days
        next_month = fail_time + pd.Timedelta(days=30)
        machine_maint = maint[(maint['machineID'] == machine_id) &
                              (maint['datetime'] > fail_time) &
                              (maint['datetime'] <= next_month)]

        for _, maint_row in machine_maint.iterrows():
            maint_comp = int(maint_row['comp'].replace('comp', '')) - 1
            fail_maint_counts[fail_comp, maint_comp] += 1

    # Normalize
    fail_maint_norm = fail_maint_counts / fail_maint_counts.sum(axis=1, keepdims=True)
    fail_maint_norm = np.nan_to_num(fail_maint_norm)

    sns.heatmap(fail_maint_norm, annot=True, fmt='.2f', cmap='YlGn',
                xticklabels=['Maint 1', 'Maint 2', 'Maint 3', 'Maint 4'],
                yticklabels=['Fail 1', 'Fail 2', 'Fail 3', 'Fail 4'],
                ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Maintenance After Each Failure\n(Normalized Frequency)',
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('Maintenance Component')
    ax3.set_ylabel('Failed Component')

    # === Panel D: Key Statistics ===
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')

    # Calculate key statistics
    total_errors = len(errors)
    total_failures = len(failures)
    total_maint = len(maint)

    # Error rate
    error_rate = total_errors / 100 / 365  # errors per machine per day
    failure_rate = total_failures / 100 / 365  # failures per machine per day

    # Most common sequences
    stats_text = f"""
KEY STATISTICS

Data Scope:
  • 100 machines monitored
  • 365 days of operation
  • 4 sensor types (hourly)

Event Frequencies:
  • {error_rate:.2f} errors/machine/day
  • {failure_rate:.3f} failures/machine/day
  • {total_maint/100:.1f} maint events/machine

Relationships Found:
  • Error 3 → Comp 2 failures
  • Error 2 → Comp 2 failures
  • Error 1 → Comp 1 failures
  • Failures trigger same-comp maint
    """

    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                     edgecolor='#bdc3c7', linewidth=1))

    # === Panel E: Sensor Impact Summary ===
    ax5 = fig.add_subplot(gs[2, 0])

    if processed_df is not None:
        sensor_cols = ['volt_mean', 'rotate_mean', 'pressure_mean', 'vibration_mean']
        failure_cols = ['failure_comp1', 'failure_comp2', 'failure_comp3', 'failure_comp4']

        corrs = []
        for sensor in sensor_cols:
            sensor_corrs = []
            for failure in failure_cols:
                corr = processed_df[sensor].corr(processed_df[failure])
                if not np.isnan(corr):
                    sensor_corrs.append(abs(corr))
            if sensor_corrs:
                corrs.append(np.mean(sensor_corrs))
            else:
                corrs.append(0.01)  # Default small value

        # Ensure we have valid data
        max_corr = max(corrs) if corrs and max(corrs) > 0 else 0.1

        colors = ['#3498db', '#e67e22', '#27ae60', '#9b59b6']
        bars = ax5.bar(['Voltage', 'Rotation', 'Pressure', 'Vibration'],
                       corrs, color=colors, edgecolor='white', linewidth=2)
        ax5.set_ylabel('Avg |Correlation| with Failures', fontsize=10)
        ax5.set_title('Sensor Impact on Failures', fontsize=11, fontweight='bold')
        ax5.set_ylim(0, max_corr * 1.3)

        for bar, val in zip(bars, corrs):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    # === Panel F: Error Predictive Value ===
    ax6 = fig.add_subplot(gs[2, 1])

    # Calculate how often each error type precedes a failure
    error_predictive = {}
    for i in range(1, 6):
        error_name = f'error{i}'
        errors_of_type = errors[errors['errorID'] == error_name]

        preceded_failure = 0
        for _, err_row in errors_of_type.sample(min(200, len(errors_of_type))).iterrows():
            machine_id = err_row['machineID']
            err_time = err_row['datetime']

            # Check for failure in next 7 days
            week_after = err_time + pd.Timedelta(days=7)
            future_failures = failures[(failures['machineID'] == machine_id) &
                                       (failures['datetime'] > err_time) &
                                       (failures['datetime'] <= week_after)]

            if len(future_failures) > 0:
                preceded_failure += 1

        error_predictive[f'Error {i}'] = preceded_failure / min(200, len(errors_of_type)) * 100

    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#9b59b6']
    bars = ax6.bar(error_predictive.keys(), error_predictive.values(),
                   color=colors, edgecolor='white', linewidth=2)
    ax6.set_ylabel('% Followed by Failure (7 days)', fontsize=10)
    ax6.set_title('Error Predictive Value', fontsize=11, fontweight='bold')
    max_pred = max(error_predictive.values()) if error_predictive.values() else 10
    ax6.set_ylim(0, max(max_pred * 1.3, 1))

    for bar, val in zip(bars, error_predictive.values()):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # === Panel G: Summary Box ===
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    summary_text = """
SUMMARY: How Variables Affect Each Other

1. SENSORS → ERRORS
   Abnormal sensor readings trigger
   error codes in the system

2. ERRORS → FAILURES
   Error codes are predictive signals
   for upcoming component failures

3. FAILURES → MAINTENANCE
   Component failures trigger
   maintenance actions (repairs)

4. KEY INSIGHT
   Error 3 and Error 2 are the
   strongest predictors of failure

This causal chain enables:
• Early warning systems
• Predictive maintenance
• Reduced downtime
    """

    ax7.text(0.1, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9e79f',
                     edgecolor='#f1c40f', linewidth=2))

    plt.savefig(output_dir / 'interaction_summary.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: interaction_summary.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    # Load all data
    print("\nLoading data...")
    telemetry, errors, failures, machines, maint, processed_df = load_all_data()

    # Generate all figures
    create_dataset_overview(telemetry, errors, failures, machines, maint)
    create_correlation_matrix(processed_df)
    create_time_series_patterns(telemetry)
    create_event_analysis(errors, failures, maint)
    create_sensor_analysis(telemetry, failures)
    create_interaction_summary(processed_df, errors, failures, maint)

    print("\n" + "=" * 70)
    print("ALL DATASET FIGURES GENERATED!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")

    print("\n" + "-" * 70)
    print("FIGURE DESCRIPTIONS:")
    print("-" * 70)
    print("""
1. dataset_overview.png
   - Complete overview of all data sources
   - Distributions of sensors, errors, failures, maintenance
   - Machine characteristics (age, model)
   - Monthly event timeline

2. correlation_matrix.png
   - Full variable correlation heatmap
   - Focused error-failure-maintenance correlations
   - Shows which variables move together

3. time_series_patterns.png
   - Temporal patterns in sensor data
   - Daily and weekly trends
   - Moving averages

4. event_analysis.png
   - Errors before failures
   - Time from failure to maintenance
   - Component failure distribution
   - PdM event chain visualization

5. sensor_analysis.png
   - Sensor value distributions
   - Pre-failure vs normal readings
   - Sensor relationships
   - Machine variability analysis

6. interaction_summary.png
   - Complete causal flow diagram
   - Error-failure relationship matrix
   - Failure-maintenance relationship matrix
   - Key statistics and insights
""")
