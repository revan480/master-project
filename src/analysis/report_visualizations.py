"""
project Visualizations Module

Generates publication-quality figures for Master's project on
Causal Discovery for Predictive Maintenance.

Figures Generated:
1. Algorithm Comparison Bar Chart (edge counts, execution times)
2. Stability Score Heatmap (edges x algorithms)
3. Algorithm Agreement Matrix (Jaccard similarity)
4. High-Confidence Causal Graph Network
5. Edge Stability Distribution (histogram per algorithm)
6. Consensus Edges Across Algorithms
7. Variable-wise Causal Relationships
8. Execution Time Comparison
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})


class projectVisualizer:
    """Generate project-quality visualizations for causal discovery results."""

    def __init__(self, results_path: str = 'results'):
        self.results_path = Path(results_path)
        self.graphs_path = self.results_path / 'causal_graphs'
        self.stability_path = self.results_path / 'stability_scores'
        self.figures_path = self.results_path / 'figures'
        self.figures_path.mkdir(parents=True, exist_ok=True)

        # Load all data
        self.machine_results = {}
        self.stability_data = {}
        self.algorithms = []
        self.variables = []

        self._load_all_data()

    def _load_all_data(self):
        """Load all results and stability data."""
        print("Loading all data for visualization...")

        # Load causal graph results
        for file_path in sorted(self.graphs_path.glob("machine_*.json")):
            machine_id = int(file_path.stem.split('_')[1])
            with open(file_path) as f:
                self.machine_results[machine_id] = json.load(f)

        # Get algorithms and variables from first result
        if self.machine_results:
            first_result = list(self.machine_results.values())[0]
            self.algorithms = list(first_result.get('algorithms', {}).keys())
            for alg_data in first_result.get('algorithms', {}).values():
                if alg_data.get('variables'):
                    self.variables = alg_data['variables']
                    break

        # Load stability scores
        for alg in self.algorithms:
            alg_file = alg.replace('+', 'plus')
            stability_file = self.stability_path / f"stability_{alg_file}.csv"
            if stability_file.exists():
                self.stability_data[alg] = pd.read_csv(stability_file)

        # Load agreement matrix
        agreement_file = self.stability_path / "algorithm_agreement.csv"
        if agreement_file.exists():
            self.agreement_matrix = pd.read_csv(agreement_file, index_col=0)
        else:
            self.agreement_matrix = None

        # Load high confidence edges
        high_conf_file = self.stability_path / "high_confidence_edges.csv"
        if high_conf_file.exists():
            self.high_confidence_edges = pd.read_csv(high_conf_file)
        else:
            self.high_confidence_edges = pd.DataFrame()

        print(f"Loaded: {len(self.machine_results)} machines, {len(self.algorithms)} algorithms")

    def figure1_algorithm_comparison(self):
        """
        Figure 1: Algorithm Comparison
        - Bar chart showing edge counts and execution times per algorithm
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Collect statistics
        alg_stats = defaultdict(lambda: {'edges': [], 'times': []})

        for machine_id, results in self.machine_results.items():
            for alg, data in results.get('algorithms', {}).items():
                adj_matrix = np.array(data.get('adjacency_matrix', []))
                if adj_matrix.size > 0:
                    edge_count = int(np.sum(adj_matrix))
                    alg_stats[alg]['edges'].append(edge_count)
                exec_time = data.get('execution_time', 0)
                alg_stats[alg]['times'].append(exec_time)

        # Prepare data
        alg_names = list(alg_stats.keys())
        mean_edges = [np.mean(alg_stats[alg]['edges']) for alg in alg_names]
        std_edges = [np.std(alg_stats[alg]['edges']) for alg in alg_names]
        mean_times = [np.mean(alg_stats[alg]['times']) for alg in alg_names]
        std_times = [np.std(alg_stats[alg]['times']) for alg in alg_names]

        # Sort by mean edges
        sorted_idx = np.argsort(mean_edges)[::-1]
        alg_names = [alg_names[i] for i in sorted_idx]
        mean_edges = [mean_edges[i] for i in sorted_idx]
        std_edges = [std_edges[i] for i in sorted_idx]
        mean_times = [mean_times[i] for i in sorted_idx]
        std_times = [std_times[i] for i in sorted_idx]

        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(alg_names)))

        # Plot 1: Edge counts
        x = np.arange(len(alg_names))
        bars1 = axes[0].bar(x, mean_edges, yerr=std_edges, capsize=3, color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_xlabel('Algorithm')
        axes[0].set_ylabel('Average Number of Edges')
        axes[0].set_title('(a) Average Edge Count per Algorithm')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(alg_names, rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars1, mean_edges):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)

        # Plot 2: Execution times
        bars2 = axes[1].bar(x, mean_times, yerr=std_times, capsize=3, color=colors, edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel('Algorithm')
        axes[1].set_ylabel('Average Execution Time (seconds)')
        axes[1].set_title('(b) Average Execution Time per Algorithm')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(alg_names, rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars2, mean_times):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}s', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig1_algorithm_comparison.png')
        plt.savefig(self.figures_path / 'fig1_algorithm_comparison.pdf')
        plt.close()
        print("Saved: fig1_algorithm_comparison.png/pdf")

    def figure2_stability_heatmap(self):
        """
        Figure 2: Stability Score Heatmap
        - Shows stability of top edges across algorithms
        """
        # Get top 30 edges by average stability across algorithms
        edge_stabilities = defaultdict(dict)

        for alg, df in self.stability_data.items():
            for _, row in df.iterrows():
                edge_key = f"{row['source']} → {row['target']}"
                edge_stabilities[edge_key][alg] = row['stability']

        # Calculate average stability and filter
        edge_avg = {edge: np.mean(list(stabs.values()))
                   for edge, stabs in edge_stabilities.items()}
        top_edges = sorted(edge_avg.keys(), key=lambda x: edge_avg[x], reverse=True)[:30]

        # Create matrix
        matrix = np.zeros((len(top_edges), len(self.algorithms)))
        for i, edge in enumerate(top_edges):
            for j, alg in enumerate(self.algorithms):
                matrix[i, j] = edge_stabilities[edge].get(alg, 0)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Custom colormap
        cmap = LinearSegmentedColormap.from_list('stability',
            ['#ffffff', '#fee0d2', '#fc9272', '#de2d26'])

        sns.heatmap(matrix, xticklabels=self.algorithms, yticklabels=top_edges,
                   cmap=cmap, annot=True, fmt='.0f', cbar_kws={'label': 'Stability Score (%)'},
                   linewidths=0.5, ax=ax, vmin=0, vmax=100)

        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Causal Edge (Source → Target)')
        ax.set_title('Edge Stability Scores Across Algorithms\n(Top 30 Edges by Average Stability)')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig2_stability_heatmap.png')
        plt.savefig(self.figures_path / 'fig2_stability_heatmap.pdf')
        plt.close()
        print("Saved: fig2_stability_heatmap.png/pdf")

    def figure3_algorithm_agreement(self):
        """
        Figure 3: Algorithm Agreement Matrix
        - Jaccard similarity between algorithm outputs
        """
        if self.agreement_matrix is None:
            print("No agreement matrix found, skipping figure 3")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        mask = np.triu(np.ones_like(self.agreement_matrix, dtype=bool), k=1)

        sns.heatmap(self.agreement_matrix, annot=True, fmt='.2f',
                   cmap='RdYlGn', center=0.5, vmin=0, vmax=1,
                   square=True, linewidths=0.5, ax=ax,
                   cbar_kws={'label': 'Jaccard Similarity'})

        ax.set_title('Pairwise Algorithm Agreement\n(Jaccard Similarity of Discovered Edges)')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Algorithm')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig3_algorithm_agreement.png')
        plt.savefig(self.figures_path / 'fig3_algorithm_agreement.pdf')
        plt.close()
        print("Saved: fig3_algorithm_agreement.png/pdf")

    def figure4_consensus_graph(self):
        """
        Figure 4: Consensus Causal Graph
        - Network visualization of high-confidence edges
        """
        try:
            import networkx as nx
        except ImportError:
            print("networkx not available, skipping figure 4")
            return

        if self.high_confidence_edges.empty:
            print("No high-confidence edges found, skipping figure 4")
            return

        # Create graph
        G = nx.DiGraph()

        # Add edges with attributes
        for _, row in self.high_confidence_edges.iterrows():
            G.add_edge(row['source'], row['target'],
                      weight=row['avg_stability'],
                      num_algorithms=row['num_algorithms'])

        # Layout
        fig, ax = plt.subplots(figsize=(14, 10))

        # Use spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Node colors by variable type
        node_colors = []
        for node in G.nodes():
            if 'volt' in node:
                node_colors.append('#e41a1c')  # Red
            elif 'rotate' in node:
                node_colors.append('#377eb8')  # Blue
            elif 'pressure' in node:
                node_colors.append('#4daf4a')  # Green
            elif 'vibration' in node:
                node_colors.append('#984ea3')  # Purple
            elif 'error' in node:
                node_colors.append('#ff7f00')  # Orange
            elif 'failure' in node:
                node_colors.append('#000000')  # Black
            else:
                node_colors.append('#999999')  # Gray

        # Edge widths by stability
        edge_widths = [G[u][v]['weight'] / 30 for u, v in G.edges()]

        # Edge colors by number of algorithms
        edge_colors = [G[u][v]['num_algorithms'] for u, v in G.edges()]

        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800,
                              alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                                       edge_cmap=plt.cm.YlOrRd,
                                       width=edge_widths, alpha=0.7,
                                       arrows=True, arrowsize=15,
                                       connectionstyle="arc3,rad=0.1", ax=ax)

        # Add colorbar for edges
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                   norm=plt.Normalize(vmin=min(edge_colors),
                                                     vmax=max(edge_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label('Number of Agreeing Algorithms')

        # Legend for node colors
        legend_elements = [
            mpatches.Patch(color='#e41a1c', label='Voltage'),
            mpatches.Patch(color='#377eb8', label='Rotation'),
            mpatches.Patch(color='#4daf4a', label='Pressure'),
            mpatches.Patch(color='#984ea3', label='Vibration'),
            mpatches.Patch(color='#ff7f00', label='Error'),
            mpatches.Patch(color='#000000', label='Failure'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', title='Variable Type')

        ax.set_title('Consensus Causal Graph\n(High-Confidence Edges: ≥80% Stability, ≥2 Algorithms)')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig4_consensus_graph.png')
        plt.savefig(self.figures_path / 'fig4_consensus_graph.pdf')
        plt.close()
        print("Saved: fig4_consensus_graph.png/pdf")

    def figure5_stability_distribution(self):
        """
        Figure 5: Stability Score Distribution
        - Histogram/violin plot per algorithm
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Collect all stability scores per algorithm
        stability_lists = []
        for alg in self.algorithms:
            if alg in self.stability_data:
                stability_lists.append(self.stability_data[alg]['stability'].values)
            else:
                stability_lists.append([])

        # Create violin plot
        parts = ax.violinplot(stability_lists, positions=range(len(self.algorithms)),
                             showmeans=True, showmedians=True)

        # Color the violins
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.algorithms)))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(self.algorithms)))
        ax.set_xticklabels(self.algorithms, rotation=45, ha='right')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Stability Score (%)')
        ax.set_title('Distribution of Edge Stability Scores per Algorithm')
        ax.set_ylim(0, 105)

        # Add horizontal lines for thresholds
        ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='High (80%)')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Medium (50%)')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig5_stability_distribution.png')
        plt.savefig(self.figures_path / 'fig5_stability_distribution.pdf')
        plt.close()
        print("Saved: fig5_stability_distribution.png/pdf")

    def figure6_edge_categories(self):
        """
        Figure 6: Edge Categories per Algorithm
        - Stacked bar showing high/medium/low stability edges
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        categories = {'High (≥80%)': [], 'Medium (50-80%)': [], 'Low (<50%)': []}

        for alg in self.algorithms:
            if alg in self.stability_data:
                df = self.stability_data[alg]
                high = len(df[df['stability'] >= 80])
                medium = len(df[(df['stability'] >= 50) & (df['stability'] < 80)])
                low = len(df[df['stability'] < 50])
            else:
                high = medium = low = 0

            categories['High (≥80%)'].append(high)
            categories['Medium (50-80%)'].append(medium)
            categories['Low (<50%)'].append(low)

        x = np.arange(len(self.algorithms))
        width = 0.6

        # Stacked bars
        bottom = np.zeros(len(self.algorithms))
        colors = ['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red

        for (label, values), color in zip(categories.items(), colors):
            ax.bar(x, values, width, bottom=bottom, label=label, color=color, edgecolor='white')
            bottom += np.array(values)

        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Number of Edges')
        ax.set_title('Edge Stability Categories per Algorithm')
        ax.set_xticks(x)
        ax.set_xticklabels(self.algorithms, rotation=45, ha='right')
        ax.legend(title='Stability Category')

        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig6_edge_categories.png')
        plt.savefig(self.figures_path / 'fig6_edge_categories.pdf')
        plt.close()
        print("Saved: fig6_edge_categories.png/pdf")

    def figure7_variable_relationships(self):
        """
        Figure 7: Variable-wise Causal Relationships
        - Heatmap showing which variables cause which
        """
        # Aggregate adjacency matrices across all machines and algorithms
        var_to_idx = {v: i for i, v in enumerate(self.variables)}
        n_vars = len(self.variables)

        # Sum of all adjacency matrices (normalized)
        total_matrix = np.zeros((n_vars, n_vars))
        count = 0

        for machine_id, results in self.machine_results.items():
            for alg, data in results.get('algorithms', {}).items():
                adj = np.array(data.get('adjacency_matrix', []))
                if adj.shape == (n_vars, n_vars):
                    total_matrix += adj
                    count += 1

        if count > 0:
            total_matrix = total_matrix / count * 100  # Percentage

        fig, ax = plt.subplots(figsize=(14, 12))

        sns.heatmap(total_matrix, xticklabels=self.variables, yticklabels=self.variables,
                   cmap='YlOrRd', annot=False, cbar_kws={'label': 'Edge Frequency (%)'},
                   ax=ax)

        ax.set_xlabel('Effect (Target Variable)')
        ax.set_ylabel('Cause (Source Variable)')
        ax.set_title('Aggregate Causal Relationships Across All Machines and Algorithms\n(Percentage of algorithm-machine combinations showing each edge)')

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig7_variable_relationships.png')
        plt.savefig(self.figures_path / 'fig7_variable_relationships.pdf')
        plt.close()
        print("Saved: fig7_variable_relationships.png/pdf")

    def figure8_execution_boxplot(self):
        """
        Figure 8: Execution Time Boxplot
        - Detailed execution time comparison
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Collect execution times
        exec_times = {alg: [] for alg in self.algorithms}

        for machine_id, results in self.machine_results.items():
            for alg, data in results.get('algorithms', {}).items():
                exec_times[alg].append(data.get('execution_time', 0))

        # Create boxplot data
        data = [exec_times[alg] for alg in self.algorithms]

        bp = ax.boxplot(data, labels=self.algorithms, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.algorithms)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Execution Time Distribution per Algorithm\n(100 machines)')
        ax.set_xticklabels(self.algorithms, rotation=45, ha='right')

        # Add mean markers
        means = [np.mean(exec_times[alg]) for alg in self.algorithms]
        ax.scatter(range(1, len(self.algorithms) + 1), means, marker='D',
                  color='red', s=50, zorder=5, label='Mean')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig8_execution_boxplot.png')
        plt.savefig(self.figures_path / 'fig8_execution_boxplot.pdf')
        plt.close()
        print("Saved: fig8_execution_boxplot.png/pdf")

    def figure9_consensus_by_threshold(self):
        """
        Figure 9: Consensus Edges by Stability Threshold
        - Shows how many edges pass different stability thresholds
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        for alg in self.algorithms:
            if alg in self.stability_data:
                df = self.stability_data[alg]
                counts = [len(df[df['stability'] >= t]) for t in thresholds]
                ax.plot(thresholds, counts, marker='o', label=alg, linewidth=2, markersize=5)

        ax.set_xlabel('Minimum Stability Threshold (%)')
        ax.set_ylabel('Number of Edges')
        ax.set_title('Number of Edges Above Stability Threshold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlim(0, 100)
        ax.set_xticks(thresholds)

        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig9_consensus_by_threshold.png')
        plt.savefig(self.figures_path / 'fig9_consensus_by_threshold.pdf')
        plt.close()
        print("Saved: fig9_consensus_by_threshold.png/pdf")

    def figure10_summary_table(self):
        """
        Figure 10: Summary Statistics Table
        - Creates a visual table with key metrics
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')

        # Prepare data
        table_data = []
        columns = ['Algorithm', 'Avg Edges', 'Std Edges', 'Avg Time (s)',
                  'High Stab.', 'Med Stab.', 'Low Stab.', 'Avg Stab. (%)']

        for alg in self.algorithms:
            # Get edge counts
            edges = []
            times = []
            for machine_id, results in self.machine_results.items():
                if alg in results.get('algorithms', {}):
                    data = results['algorithms'][alg]
                    adj = np.array(data.get('adjacency_matrix', []))
                    if adj.size > 0:
                        edges.append(int(np.sum(adj)))
                    times.append(data.get('execution_time', 0))

            # Get stability stats
            if alg in self.stability_data:
                df = self.stability_data[alg]
                high = len(df[df['stability'] >= 80])
                medium = len(df[(df['stability'] >= 50) & (df['stability'] < 80)])
                low = len(df[df['stability'] < 50])
                avg_stab = df['stability'].mean()
            else:
                high = medium = low = 0
                avg_stab = 0

            table_data.append([
                alg,
                f'{np.mean(edges):.1f}',
                f'{np.std(edges):.1f}',
                f'{np.mean(times):.2f}',
                str(high),
                str(medium),
                str(low),
                f'{avg_stab:.1f}'
            ])

        # Create table
        table = ax.table(cellText=table_data, colLabels=columns,
                        loc='center', cellLoc='center',
                        colColours=['#4472C4']*len(columns))

        # Style
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Header style
        for j in range(len(columns)):
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#D9E2F3')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')

        ax.set_title('Summary Statistics for All Causal Discovery Algorithms\n(100 Machines, 30 Variables)',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig10_summary_table.png')
        plt.savefig(self.figures_path / 'fig10_summary_table.pdf')
        plt.close()
        print("Saved: fig10_summary_table.png/pdf")

    def generate_all_figures(self):
        """Generate all project figures."""
        print("\n" + "="*60)
        print("GENERATING project VISUALIZATIONS")
        print("="*60 + "\n")

        self.figure1_algorithm_comparison()
        self.figure2_stability_heatmap()
        self.figure3_algorithm_agreement()
        self.figure4_consensus_graph()
        self.figure5_stability_distribution()
        self.figure6_edge_categories()
        self.figure7_variable_relationships()
        self.figure8_execution_boxplot()
        self.figure9_consensus_by_threshold()
        self.figure10_summary_table()

        print("\n" + "="*60)
        print(f"ALL FIGURES SAVED TO: {self.figures_path}")
        print("="*60)

        # List all generated files
        print("\nGenerated files:")
        for f in sorted(self.figures_path.glob("*")):
            print(f"  - {f.name}")

    def generate_project_statistics(self):
        """Generate key statistics for project text."""
        print("\n" + "="*60)
        print("KEY STATISTICS FOR project")
        print("="*60 + "\n")

        stats = {
            'total_machines': len(self.machine_results),
            'total_algorithms': len(self.algorithms),
            'total_variables': len(self.variables),
            'total_runs': len(self.machine_results) * len(self.algorithms),
        }

        # Per-algorithm statistics
        print("ALGORITHM PERFORMANCE:")
        print("-" * 80)
        print(f"{'Algorithm':<15} {'Avg Edges':>12} {'Avg Time':>12} {'High Stab':>12} {'Avg Stab':>12}")
        print("-" * 80)

        for alg in self.algorithms:
            edges = []
            times = []
            for machine_id, results in self.machine_results.items():
                if alg in results.get('algorithms', {}):
                    data = results['algorithms'][alg]
                    adj = np.array(data.get('adjacency_matrix', []))
                    if adj.size > 0:
                        edges.append(int(np.sum(adj)))
                    times.append(data.get('execution_time', 0))

            if alg in self.stability_data:
                df = self.stability_data[alg]
                high_stab = len(df[df['stability'] >= 80])
                avg_stab = df['stability'].mean()
            else:
                high_stab = 0
                avg_stab = 0

            print(f"{alg:<15} {np.mean(edges):>12.1f} {np.mean(times):>12.2f}s {high_stab:>12} {avg_stab:>12.1f}%")

        print("-" * 80)

        # High confidence edges summary
        print(f"\nHIGH-CONFIDENCE EDGES (≥80% stability, ≥2 algorithms):")
        print(f"Total: {len(self.high_confidence_edges)}")

        if not self.high_confidence_edges.empty:
            print("\nTop 15 most confident edges:")
            for i, (_, row) in enumerate(self.high_confidence_edges.head(15).iterrows()):
                print(f"  {i+1}. {row['source']} → {row['target']}: "
                      f"{row['avg_stability']:.1f}% ({row['num_algorithms']} algorithms)")

        # Save statistics to JSON
        stats_output = {
            'summary': stats,
            'algorithm_stats': {},
            'high_confidence_edges': self.high_confidence_edges.to_dict('records') if not self.high_confidence_edges.empty else []
        }

        for alg in self.algorithms:
            edges = []
            times = []
            for machine_id, results in self.machine_results.items():
                if alg in results.get('algorithms', {}):
                    data = results['algorithms'][alg]
                    adj = np.array(data.get('adjacency_matrix', []))
                    if adj.size > 0:
                        edges.append(int(np.sum(adj)))
                    times.append(data.get('execution_time', 0))

            if alg in self.stability_data:
                df = self.stability_data[alg]
                stats_output['algorithm_stats'][alg] = {
                    'avg_edges': float(np.mean(edges)),
                    'std_edges': float(np.std(edges)),
                    'avg_time': float(np.mean(times)),
                    'total_unique_edges': len(df),
                    'high_stability_edges': len(df[df['stability'] >= 80]),
                    'medium_stability_edges': len(df[(df['stability'] >= 50) & (df['stability'] < 80)]),
                    'low_stability_edges': len(df[df['stability'] < 50]),
                    'avg_stability': float(df['stability'].mean()),
                    'max_stability': float(df['stability'].max())
                }

        with open(self.figures_path / 'project_statistics.json', 'w') as f:
            json.dump(stats_output, f, indent=2)

        print(f"\nStatistics saved to: {self.figures_path / 'project_statistics.json'}")


def main():
    """Generate all project visualizations and statistics."""
    visualizer = projectVisualizer(results_path='results')
    visualizer.generate_all_figures()
    visualizer.generate_project_statistics()


if __name__ == '__main__':
    main()
