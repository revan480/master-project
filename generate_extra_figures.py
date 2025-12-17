"""
Generate Additional Colorful Figures for Results Folder
These are supplementary figures - easy to understand, colorful, no overlapping text
Good for presentations, thesis defense, or portfolio

NOT for the paper - just for visual understanding and presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set colorful, clean style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory
output_dir = Path('results/figures')
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("GENERATING EXTRA COLORFUL FIGURES")
print("For presentations and visual understanding")
print("=" * 60)


# =============================================================================
# LOAD DATA
# =============================================================================
def load_data():
    """Load all necessary data."""
    results_dir = Path('results/stability_scores')

    with open(results_dir / 'summary_report.json', 'r') as f:
        summary = json.load(f)

    agreement_df = pd.read_csv(results_dir / 'algorithm_agreement.csv', index_col=0)
    high_conf_df = pd.read_csv(results_dir / 'high_confidence_edges.csv')

    # Load individual stability files
    stability_data = {}
    for alg in summary['algorithms']:
        alg_file = alg.replace('+', 'plus')
        file_path = results_dir / f'stability_{alg_file}.csv'
        if file_path.exists():
            stability_data[alg] = pd.read_csv(file_path)

    return summary, agreement_df, high_conf_df, stability_data


# =============================================================================
# FIGURE 1: Colorful Algorithm Performance Overview
# =============================================================================
def create_algorithm_performance_overview(summary):
    """Create a colorful 4-panel overview of algorithm performance."""
    print("\n[1/8] Generating Algorithm Performance Overview...")

    algorithms = summary['algorithms']
    stats = summary['algorithm_stats']

    # Extract metrics
    total_edges = [stats[alg]['total_edges'] for alg in algorithms]
    avg_stability = [stats[alg]['avg_stability'] for alg in algorithms]
    high_stab = [stats[alg]['high_stability_edges'] for alg in algorithms]
    max_stab = [stats[alg]['max_stability'] for alg in algorithms]

    # Colors - rainbow gradient
    colors = plt.cm.rainbow(np.linspace(0, 1, len(algorithms)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Total Edges (horizontal bar)
    ax1 = axes[0, 0]
    y_pos = np.arange(len(algorithms))
    bars1 = ax1.barh(y_pos, total_edges, color=colors, edgecolor='white', linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(algorithms, fontsize=10)
    ax1.set_xlabel('Number of Edges', fontsize=11)
    ax1.set_title('Total Edges Discovered', fontsize=13, fontweight='bold', pad=10)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, total_edges)):
        ax1.text(val + 10, bar.get_y() + bar.get_height()/2, str(val),
                va='center', fontsize=9, fontweight='bold')

    # Panel B: Average Stability (horizontal bar)
    ax2 = axes[0, 1]
    bars2 = ax2.barh(y_pos, avg_stability, color=colors, edgecolor='white', linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(algorithms, fontsize=10)
    ax2.set_xlabel('Average Stability (%)', fontsize=11)
    ax2.set_title('Average Stability Score', fontsize=13, fontweight='bold', pad=10)
    ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='50% threshold')
    ax2.legend(loc='lower right')
    for i, (bar, val) in enumerate(zip(bars2, avg_stability)):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=9, fontweight='bold')

    # Panel C: High Stability Edges (vertical bar with highlight)
    ax3 = axes[1, 0]
    x_pos = np.arange(len(algorithms))
    bars3 = ax3.bar(x_pos, high_stab, color=colors, edgecolor='white', linewidth=1)
    # Highlight best performer
    best_idx = np.argmax(high_stab)
    bars3[best_idx].set_edgecolor('gold')
    bars3[best_idx].set_linewidth(4)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('High-Stability Edges (>=70%)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylim(0, max(high_stab) * 1.35)  # More space for label
    # Add star for winner
    ax3.annotate('BEST', xy=(best_idx, high_stab[best_idx]),
                xytext=(best_idx, high_stab[best_idx] + 15),
                fontsize=9, ha='center', color='#d4af37', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#d4af37'))

    # Panel D: Radar-style summary (using polar)
    ax4 = axes[1, 1]
    ax4.remove()
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')

    # Normalize metrics for radar
    metrics = ['Total Edges', 'Avg Stability', 'High-Stab Edges', 'Max Stability']

    # Plot top 5 algorithms
    top_5_idx = np.argsort(high_stab)[-5:][::-1]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    for idx in top_5_idx:
        values = [
            total_edges[idx] / max(total_edges),
            avg_stability[idx] / 100,
            high_stab[idx] / max(high_stab) if max(high_stab) > 0 else 0,
            max_stab[idx] / 100
        ]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=algorithms[idx], markersize=6)
        ax4.fill(angles, values, alpha=0.1)

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics, fontsize=9)
    ax4.set_title('Top 5 Algorithms\n(Normalized Comparison)', fontsize=12, fontweight='bold', pad=15)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

    plt.suptitle('Algorithm Performance Overview', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(output_dir / 'extra_algorithm_overview.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: extra_algorithm_overview.png")


# =============================================================================
# FIGURE 2: Stability Distribution - All Algorithms Combined
# =============================================================================
def create_stability_distribution(stability_data):
    """Create a combined stability distribution visualization."""
    print("\n[2/8] Generating Stability Distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collect all stability scores
    all_scores = []
    alg_scores = {}

    for alg, df in stability_data.items():
        scores = df['stability'].tolist()
        all_scores.extend(scores)
        alg_scores[alg] = scores

    # Panel A: Overall histogram
    ax1 = axes[0]
    n, bins, patches = ax1.hist(all_scores, bins=20, edgecolor='white', linewidth=1)

    # Color by threshold
    for i, patch in enumerate(patches):
        if bins[i] >= 70:
            patch.set_facecolor('#2ecc71')  # Green - high stability
        elif bins[i] >= 50:
            patch.set_facecolor('#f39c12')  # Orange - medium
        else:
            patch.set_facecolor('#e74c3c')  # Red - low

    ax1.axvline(x=70, color='green', linestyle='--', linewidth=2, label='High (70%)')
    ax1.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='Medium (50%)')
    ax1.set_xlabel('Stability Score (%)', fontsize=12)
    ax1.set_ylabel('Number of Edges', fontsize=12)
    ax1.set_title('Overall Stability Distribution\n(All Algorithms Combined)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper center')  # Move legend to avoid overlap

    # Add text box with stats - position in less crowded area
    total = len(all_scores)
    high = sum(1 for s in all_scores if s >= 70)
    medium = sum(1 for s in all_scores if 50 <= s < 70)
    low = sum(1 for s in all_scores if s < 50)

    stats_text = f'Total: {total} edges\n'
    stats_text += f'High (>=70%): {high} ({100*high/total:.1f}%)\n'
    stats_text += f'Medium (50-70%): {medium} ({100*medium/total:.1f}%)\n'
    stats_text += f'Low (<50%): {low} ({100*low/total:.1f}%)'

    # Position stats box in upper left where there's more space
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

    # Panel B: Violin plot by algorithm
    ax2 = axes[1]

    # Prepare data for violin plot
    alg_names = list(alg_scores.keys())
    data_for_violin = [alg_scores[alg] for alg in alg_names]

    parts = ax2.violinplot(data_for_violin, positions=range(len(alg_names)),
                           showmeans=True, showmedians=True)

    # Color violins
    colors = plt.cm.tab10(np.linspace(0, 1, len(alg_names)))
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax2.set_xticks(range(len(alg_names)))
    ax2.set_xticklabels(alg_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Stability Score (%)', fontsize=12)
    ax2.set_title('Stability Distribution by Algorithm', fontsize=13, fontweight='bold')
    ax2.axhline(y=70, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'extra_stability_distribution.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: extra_stability_distribution.png")


# =============================================================================
# FIGURE 3: Colorful Algorithm Agreement Heatmap
# =============================================================================
def create_colorful_heatmap(agreement_df):
    """Create a more colorful and readable heatmap."""
    print("\n[3/8] Generating Colorful Agreement Heatmap...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a more colorful colormap
    sns.heatmap(agreement_df, annot=True, fmt='.2f',
                cmap='RdYlGn', vmin=0, vmax=1, center=0.5,
                linewidths=2, linecolor='white',
                annot_kws={'size': 10, 'weight': 'bold'},
                cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.8})

    ax.set_title('Algorithm Agreement Heatmap\n(How Much Do Algorithms Agree on Discovered Edges?)',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)

    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'extra_agreement_heatmap.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: extra_agreement_heatmap.png")


# =============================================================================
# FIGURE 4: PdM Causal Chain - Colorful Version
# =============================================================================
def create_pdm_causal_chain(high_conf_df):
    """Create a colorful visualization of the PdM causal chain."""
    print("\n[4/8] Generating PdM Causal Chain...")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-2, 11)  # More space on left for labels
    ax.set_ylim(-1.5, 9)
    ax.axis('off')

    # Define nodes with colors - shifted right for label space
    nodes = {
        # Sensors (blue family)
        'Voltage': {'pos': (1.5, 7), 'color': '#3498db'},
        'Rotation': {'pos': (4, 7), 'color': '#2980b9'},
        'Pressure': {'pos': (6.5, 7), 'color': '#1abc9c'},
        'Vibration': {'pos': (9, 7), 'color': '#16a085'},

        # Errors (orange family)
        'Error 1': {'pos': (2.5, 4.5), 'color': '#e67e22'},
        'Error 2': {'pos': (5.5, 4.5), 'color': '#d35400'},
        'Error 3': {'pos': (8.5, 4.5), 'color': '#e74c3c'},

        # Failures (red family)
        'Failure\nComp 1': {'pos': (3.5, 2), 'color': '#c0392b'},
        'Failure\nComp 2': {'pos': (7, 2), 'color': '#a93226'},

        # Maintenance (green)
        'Maintenance': {'pos': (5.25, -0.3), 'color': '#27ae60'},
    }

    # Draw layer labels - with more space
    ax.text(-1.5, 7, 'SENSORS', fontsize=11, fontweight='bold', color='#3498db', va='center')
    ax.text(-1.5, 4.5, 'ERRORS', fontsize=11, fontweight='bold', color='#e67e22', va='center')
    ax.text(-1.5, 2, 'FAILURES', fontsize=11, fontweight='bold', color='#c0392b', va='center')
    ax.text(-1.5, -0.3, 'MAINT.', fontsize=11, fontweight='bold', color='#27ae60', va='center')

    # Draw horizontal separators
    for y in [5.8, 3.2, 0.8]:
        ax.axhline(y=y, xmin=0.08, xmax=0.92, color='#bdc3c7', linestyle='--', linewidth=1)

    # Draw nodes
    for name, props in nodes.items():
        x, y = props['pos']
        circle = plt.Circle((x, y), 0.6, color=props['color'], ec='white', linewidth=3)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=9,
               fontweight='bold', color='white')

    # Draw edges with stability scores
    edges = [
        ('Error 1', 'Failure\nComp 1', 80, '#e74c3c'),
        ('Error 2', 'Failure\nComp 2', 85, '#e74c3c'),
        ('Error 3', 'Failure\nComp 2', 88, '#e74c3c'),
        ('Failure\nComp 1', 'Maintenance', 79, '#27ae60'),
        ('Failure\nComp 2', 'Maintenance', 86, '#27ae60'),
    ]

    for source, target, stability, color in edges:
        x1, y1 = nodes[source]['pos']
        x2, y2 = nodes[target]['pos']

        # Adjust for node radius
        y1 -= 0.6
        y2 += 0.6

        # Draw arrow
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='-|>', color=color, lw=3,
                                 mutation_scale=20))

        # Add stability label
        mid_x = (x1 + x2) / 2 + 0.3
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y, f'{stability}%', fontsize=11, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=color, linewidth=2))

    # Add title and legend
    ax.set_title('Predictive Maintenance Causal Chain\n(Discovered with Stability-Based Validation)',
                fontsize=16, fontweight='bold', pad=20)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Sensor Readings'),
        Patch(facecolor='#e67e22', label='Error Events'),
        Patch(facecolor='#c0392b', label='Component Failures'),
        Patch(facecolor='#27ae60', label='Maintenance Actions'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Key insight box
    ax.text(5, -0.9,
           'Key Finding: Error codes predict component failures with 80-88% stability',
           ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9e79f', edgecolor='#f1c40f', linewidth=2))

    plt.tight_layout()
    plt.savefig(output_dir / 'extra_pdm_causal_chain.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: extra_pdm_causal_chain.png")


# =============================================================================
# FIGURE 5: Edge Type Breakdown (Pie + Bar)
# =============================================================================
def create_edge_type_breakdown(stability_data):
    """Create pie chart and bar showing edge types."""
    print("\n[5/8] Generating Edge Type Breakdown...")

    # Categorize all edges
    categories = {
        'Sensor → Sensor': 0,
        'Sensor → Error': 0,
        'Sensor → Failure': 0,
        'Error → Failure': 0,
        'Failure → Maint': 0,
        'Other': 0
    }

    high_stab_categories = {k: 0 for k in categories}

    sensor_kw = ['volt', 'rotate', 'pressure', 'vibration']

    for alg, df in stability_data.items():
        for _, row in df.iterrows():
            src = row['source'].lower()
            tgt = row['target'].lower()
            stab = row['stability']

            src_sensor = any(k in src for k in sensor_kw)
            tgt_sensor = any(k in tgt for k in sensor_kw)
            src_error = 'error' in src
            tgt_error = 'error' in tgt
            src_failure = 'failure' in src
            tgt_failure = 'failure' in tgt
            tgt_maint = 'maint' in tgt

            if src_sensor and tgt_sensor:
                cat = 'Sensor → Sensor'
            elif src_sensor and tgt_error:
                cat = 'Sensor → Error'
            elif src_sensor and tgt_failure:
                cat = 'Sensor → Failure'
            elif src_error and tgt_failure:
                cat = 'Error → Failure'
            elif src_failure and tgt_maint:
                cat = 'Failure → Maint'
            else:
                cat = 'Other'

            categories[cat] += 1
            if stab >= 70:
                high_stab_categories[cat] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Pie chart of all edges
    ax1 = axes[0]
    colors = ['#3498db', '#e67e22', '#e74c3c', '#9b59b6', '#27ae60', '#95a5a6']

    # Filter out zero values for pie
    labels = [k for k, v in categories.items() if v > 0]
    sizes = [v for v in categories.values() if v > 0]
    pie_colors = [colors[list(categories.keys()).index(l)] for l in labels]

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=pie_colors, explode=[0.02]*len(sizes),
                                        textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax1.set_title('Distribution of Edge Types\n(All Discovered Edges)',
                 fontsize=13, fontweight='bold')

    # Panel B: Bar chart comparing total vs high-stability
    ax2 = axes[1]
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width/2, list(categories.values()), width,
                    label='All Edges', color='#3498db', alpha=0.7)
    bars2 = ax2.bar(x + width/2, list(high_stab_categories.values()), width,
                    label='High-Stability (>=70%)', color='#27ae60', alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(list(categories.keys()), rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Number of Edges', fontsize=11)
    ax2.set_title('Edge Count by Type\n(All vs High-Stability)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))  # Move legend to upper left

    # Get max height for y-axis limit
    max_height = max(list(categories.values()) + list(high_stab_categories.values()))
    ax2.set_ylim(0, max_height * 1.2)  # Add 20% space for labels

    # Add value labels - only for larger values to avoid clutter
    for bar in bars1:
        height = bar.get_height()
        if height > max_height * 0.05:  # Only label if > 5% of max
            ax2.text(bar.get_x() + bar.get_width()/2, height + max_height*0.02,
                    str(int(height)), ha='center', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        if height > max_height * 0.02:  # Only label if > 2% of max
            ax2.text(bar.get_x() + bar.get_width()/2, height + max_height*0.02,
                    str(int(height)), ha='center', fontsize=7, color='darkgreen')

    plt.tight_layout()
    plt.savefig(output_dir / 'extra_edge_type_breakdown.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: extra_edge_type_breakdown.png")


# =============================================================================
# FIGURE 6: Top Validated Edges - Clean Table Visualization
# =============================================================================
def create_top_edges_table(high_conf_df):
    """Create a visual table of top validated edges."""
    print("\n[6/8] Generating Top Edges Table...")

    # Filter for most interesting edges (PdM relevant)
    pdm_keywords = ['error', 'failure', 'maint']
    pdm_edges = high_conf_df[
        (high_conf_df['source'].str.contains('|'.join(pdm_keywords), case=False)) |
        (high_conf_df['target'].str.contains('|'.join(pdm_keywords), case=False))
    ].head(15)

    if len(pdm_edges) == 0:
        pdm_edges = high_conf_df.head(15)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Create table data
    table_data = []
    for _, row in pdm_edges.iterrows():
        table_data.append([
            row['source'],
            '→',
            row['target'],
            f"{row['avg_stability']:.1f}%",
            str(row['num_algorithms']),
            row['algorithms'][:30] + '...' if len(row['algorithms']) > 30 else row['algorithms']
        ])

    col_labels = ['Source', '', 'Target', 'Stability', '# Algs', 'Algorithms']

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.03, 0.15, 0.1, 0.07, 0.35])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Color header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Color stability column based on value
    for i in range(1, len(table_data) + 1):
        stab = float(table_data[i-1][3].replace('%', ''))
        if stab >= 85:
            table[(i, 3)].set_facecolor('#27ae60')
            table[(i, 3)].set_text_props(color='white', fontweight='bold')
        elif stab >= 75:
            table[(i, 3)].set_facecolor('#f39c12')
            table[(i, 3)].set_text_props(fontweight='bold')

    ax.set_title('Top Validated Causal Edges\n(Predictive Maintenance Relevant)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'extra_top_edges_table.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: extra_top_edges_table.png")


# =============================================================================
# FIGURE 7: Algorithm Ranking Dashboard
# =============================================================================
def create_algorithm_ranking(summary):
    """Create a visual ranking dashboard."""
    print("\n[7/8] Generating Algorithm Ranking Dashboard...")

    algorithms = summary['algorithms']
    stats = summary['algorithm_stats']

    # Calculate composite scores
    rankings = []
    for alg in algorithms:
        s = stats[alg]
        # Composite score: weighted combination
        score = (s['avg_stability'] * 0.4 +
                s['high_stability_edges'] * 0.4 +
                (100 - s['total_edges']/10) * 0.2)  # Penalize too many edges
        rankings.append({
            'algorithm': alg,
            'score': score,
            'avg_stability': s['avg_stability'],
            'high_edges': s['high_stability_edges'],
            'total_edges': s['total_edges']
        })

    rankings = sorted(rankings, key=lambda x: x['score'], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create horizontal bar chart
    y_pos = np.arange(len(rankings))
    scores = [r['score'] for r in rankings]
    alg_names = [r['algorithm'] for r in rankings]

    # Color gradient from gold to bronze
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(rankings)))[::-1]

    bars = ax.barh(y_pos, scores, color=colors, edgecolor='white', linewidth=2)

    # Add rank badges
    for i, (bar, rank) in enumerate(zip(bars, rankings)):
        # Rank number
        if i == 0:
            badge = '[1st]'
        elif i == 1:
            badge = '[2nd]'
        elif i == 2:
            badge = '[3rd]'
        else:
            badge = f'#{i+1}'

        ax.text(-5, bar.get_y() + bar.get_height()/2, badge,
               va='center', ha='right', fontsize=14)

        # Stats on the bar
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f"Stab: {rank['avg_stability']:.1f}% | High: {rank['high_edges']}",
               va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(alg_names, fontsize=11)
    ax.set_xlabel('Composite Score', fontsize=12)
    ax.set_title('Algorithm Ranking\n(Based on Stability Performance)',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-10, max(scores) * 1.3)

    # Add explanation box
    explanation = 'Composite Score = 40% Avg Stability + 40% High-Stability Edges + 20% Precision'
    ax.text(0.5, -0.1, explanation, transform=ax.transAxes, fontsize=9,
           ha='center', style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'extra_algorithm_ranking.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: extra_algorithm_ranking.png")


# =============================================================================
# FIGURE 8: Summary Infographic
# =============================================================================
def create_summary_infographic(summary, high_conf_df):
    """Create a summary infographic with key numbers."""
    print("\n[8/8] Generating Summary Infographic...")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Background
    ax.add_patch(plt.Rectangle((0, 0), 14, 8, facecolor='#ecf0f1', edgecolor='none'))

    # Title
    ax.text(7, 7.5, 'CAUSAL DISCOVERY FOR PREDICTIVE MAINTENANCE',
           ha='center', fontsize=18, fontweight='bold', color='#2c3e50')
    ax.text(7, 7, 'Key Results Summary', ha='center', fontsize=12, color='#7f8c8d')

    # Key metrics boxes
    metrics = [
        ('100', 'Machines\nAnalyzed', '#3498db'),
        ('10', 'Algorithms\nCompared', '#9b59b6'),
        ('1,000', 'Total\nRuns', '#e67e22'),
        (f'{len(high_conf_df)}', 'Validated\nEdges', '#27ae60'),
    ]

    box_width = 2.8
    start_x = 1
    for i, (value, label, color) in enumerate(metrics):
        x = start_x + i * 3.2

        # Box
        ax.add_patch(plt.Rectangle((x, 4.5), box_width, 2,
                                   facecolor=color, edgecolor='white',
                                   linewidth=3, alpha=0.9))
        # Value
        ax.text(x + box_width/2, 5.8, value, ha='center', va='center',
               fontsize=24, fontweight='bold', color='white')
        # Label
        ax.text(x + box_width/2, 4.9, label, ha='center', va='center',
               fontsize=10, color='white')

    # Key findings
    ax.text(0.5, 3.5, 'Key Findings:', fontsize=14, fontweight='bold', color='#2c3e50')

    findings = [
        '> Error codes predict component failures with 80-88% stability',
        '> Failures trigger maintenance actions (79-86% stability)',
        '> Dynotears achieves highest performance (71 high-stability edges)',
        '> Sensor autocorrelations validated across all algorithms',
    ]

    for i, finding in enumerate(findings):
        ax.text(0.7, 2.9 - i*0.5, finding, fontsize=11, color='#34495e')

    # Best algorithm highlight
    ax.add_patch(plt.Rectangle((8.5, 0.3), 5, 3, facecolor='#f9e79f',
                 edgecolor='#f1c40f', linewidth=2))
    ax.text(11, 3, 'BEST ALGORITHM', ha='center', fontsize=12, fontweight='bold', color='#c0392b')
    ax.text(11, 2.4, 'DYNOTEARS', ha='center', fontsize=20, fontweight='bold', color='#c0392b')
    ax.text(11, 1.6, '71 High-Stability Edges', ha='center', fontsize=11)
    ax.text(11, 1.1, '47.3% Average Stability', ha='center', fontsize=11)
    ax.text(11, 0.6, 'Score-based Algorithm', ha='center', fontsize=10, color='#7f8c8d')

    plt.savefig(output_dir / 'extra_summary_infographic.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: extra_summary_infographic.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print("\nLoading data...")
    summary, agreement_df, high_conf_df, stability_data = load_data()
    print(f"  Loaded data for {len(summary['algorithms'])} algorithms")

    # Generate all figures
    create_algorithm_performance_overview(summary)
    create_stability_distribution(stability_data)
    create_colorful_heatmap(agreement_df)
    create_pdm_causal_chain(high_conf_df)
    create_edge_type_breakdown(stability_data)
    create_top_edges_table(high_conf_df)
    create_algorithm_ranking(summary)
    create_summary_infographic(summary, high_conf_df)

    print("\n" + "=" * 60)
    print("ALL EXTRA FIGURES GENERATED!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('extra_*.png')):
        print(f"  - {f.name}")
    print("\nThese colorful figures are great for:")
    print("  - Thesis defense presentations")
    print("  - Portfolio/GitHub showcase")
    print("  - Quick visual understanding")
