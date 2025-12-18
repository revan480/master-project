"""
Visualization Module

Creates visualizations for causal discovery results and stability analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import click


class ResultsPlotter:
    """
    Creates visualizations for causal discovery stability analysis.
    """

    def __init__(self, results_path: str):
        """
        Initialize the ResultsPlotter.

        Args:
            results_path: Path to results directory
        """
        self.results_path = Path(results_path)
        self.stability_path = self.results_path / 'stability_scores'
        self.figures_path = self.results_path / 'figures'
        self.figures_path.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def load_stability_data(self, algorithm: str) -> pd.DataFrame:
        """Load stability data for an algorithm."""
        file_path = self.stability_path / f"stability_{algorithm.replace('+', 'plus')}.csv"

        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            return pd.DataFrame()

    def load_all_stability_data(self) -> Dict[str, pd.DataFrame]:
        """Load stability data for all algorithms."""
        data = {}

        for file_path in self.stability_path.glob("stability_*.csv"):
            algorithm = file_path.stem.replace('stability_', '').replace('plus', '+')
            data[algorithm] = pd.read_csv(file_path)

        return data

    def plot_stability_distribution(self, save: bool = True) -> plt.Figure:
        """
        Plot distribution of stability scores across algorithms.

        Args:
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        all_data = self.load_all_stability_data()

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for idx, (algorithm, df) in enumerate(all_data.items()):
            if idx >= 10:
                break

            ax = axes[idx]

            if not df.empty:
                ax.hist(df['stability'], bins=20, edgecolor='white', alpha=0.7)
                ax.axvline(df['stability'].mean(), color='red', linestyle='--',
                          label=f'Mean: {df["stability"].mean():.1f}%')
                ax.axvline(80, color='green', linestyle=':', label='80% threshold')

            ax.set_title(algorithm, fontsize=12, fontweight='bold')
            ax.set_xlabel('Stability (%)')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.set_xlim(0, 100)

        # Hide empty axes
        for idx in range(len(all_data), 10):
            axes[idx].set_visible(False)

        plt.suptitle('Stability Score Distribution by Algorithm', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            fig.savefig(self.figures_path / 'stability_distribution.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {self.figures_path / 'stability_distribution.png'}")

        return fig

    def plot_algorithm_agreement_heatmap(self, save: bool = True) -> plt.Figure:
        """
        Plot heatmap of algorithm agreement (Jaccard similarity).

        Args:
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        agreement_path = self.stability_path / 'algorithm_agreement.csv'

        if not agreement_path.exists():
            print("Algorithm agreement file not found")
            return None

        agreement_df = pd.read_csv(agreement_path, index_col=0)

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(agreement_df, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=0, vmax=1, center=0.5, ax=ax,
                   cbar_kws={'label': 'Jaccard Similarity'})

        ax.set_title('Algorithm Agreement Matrix\n(Jaccard Similarity of Discovered Edges)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Algorithm')

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_path / 'algorithm_agreement.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {self.figures_path / 'algorithm_agreement.png'}")

        return fig

    def plot_top_edges_comparison(self, top_n: int = 15, save: bool = True) -> plt.Figure:
        """
        Compare top stable edges across algorithms.

        Args:
            top_n: Number of top edges to show
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        all_data = self.load_all_stability_data()

        # Collect top edges from each algorithm
        edge_stability = {}

        for algorithm, df in all_data.items():
            if df.empty:
                continue

            for _, row in df.head(top_n * 2).iterrows():
                edge = f"{row['source']} -> {row['target']}"
                if edge not in edge_stability:
                    edge_stability[edge] = {}
                edge_stability[edge][algorithm] = row['stability']

        # Create matrix
        edges = list(edge_stability.keys())
        algorithms = list(all_data.keys())

        matrix = np.zeros((len(edges), len(algorithms)))
        for i, edge in enumerate(edges):
            for j, alg in enumerate(algorithms):
                matrix[i, j] = edge_stability[edge].get(alg, 0)

        # Sort by average stability
        avg_stability = matrix.mean(axis=1)
        sorted_idx = np.argsort(avg_stability)[::-1][:top_n]

        matrix = matrix[sorted_idx]
        edges = [edges[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.4)))

        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=algorithms, yticklabels=edges, ax=ax,
                   cbar_kws={'label': 'Stability (%)'})

        ax.set_title(f'Top {top_n} Causal Edges by Stability\n(Cross-Algorithm Comparison)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Causal Edge')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save:
            fig.savefig(self.figures_path / 'top_edges_comparison.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {self.figures_path / 'top_edges_comparison.png'}")

        return fig

    def plot_consensus_graph(self, min_stability: float = 70.0,
                             min_algorithms: int = 3,
                             save: bool = True) -> plt.Figure:
        """
        Plot consensus causal graph showing high-confidence edges.

        Args:
            min_stability: Minimum stability percentage
            min_algorithms: Minimum number of algorithms agreeing
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        high_conf_path = self.stability_path / 'high_confidence_edges.csv'

        if not high_conf_path.exists():
            print("High confidence edges file not found")
            return None

        df = pd.read_csv(high_conf_path)

        if df.empty:
            print("No high-confidence edges found")
            return None

        # Filter edges
        filtered = df[(df['avg_stability'] >= min_stability) &
                      (df['num_algorithms'] >= min_algorithms)]

        if filtered.empty:
            print(f"No edges meet criteria (stability >= {min_stability}%, algorithms >= {min_algorithms})")
            return None

        # Create graph
        G = nx.DiGraph()

        for _, row in filtered.iterrows():
            # Parse source to get base variable name
            source = row['source'].split('_lag')[0] if '_lag' in row['source'] else row['source']
            target = row['target']

            G.add_edge(source, target,
                      weight=row['avg_stability'],
                      algorithms=row['num_algorithms'])

        fig, ax = plt.subplots(figsize=(16, 12))

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw nodes
        node_sizes = [1500 + G.degree(node) * 200 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color='lightblue', edgecolors='black',
                              linewidths=2, ax=ax)

        # Draw edges with varying width based on stability
        edges = G.edges(data=True)
        edge_weights = [d['weight'] / 20 for _, _, d in edges]
        edge_colors = [d['weight'] for _, _, d in edges]

        edges_drawn = nx.draw_networkx_edges(G, pos,
                                             width=edge_weights,
                                             edge_color=edge_colors,
                                             edge_cmap=plt.cm.Reds,
                                             edge_vmin=50, edge_vmax=100,
                                             arrows=True, arrowsize=20,
                                             connectionstyle='arc3,rad=0.1',
                                             ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds,
                                   norm=plt.Normalize(vmin=50, vmax=100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Stability (%)', fontsize=12)

        ax.set_title(f'Consensus Causal Graph\n(Stability >= {min_stability}%, '
                    f'Algorithms >= {min_algorithms})', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_path / 'consensus_graph.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {self.figures_path / 'consensus_graph.png'}")

        return fig

    def plot_edge_stability_by_type(self, save: bool = True) -> plt.Figure:
        """
        Plot stability scores grouped by edge type (sensor→failure, etc.).

        Args:
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        all_data = self.load_all_stability_data()

        # Categorize edges
        categories = {
            'Sensor → Failure': [],
            'Sensor → Error': [],
            'Error → Failure': [],
            'Sensor → Sensor': [],
            'Other': []
        }

        sensor_keywords = ['volt', 'rotate', 'pressure', 'vibration']

        for algorithm, df in all_data.items():
            for _, row in df.iterrows():
                source = row['source'].lower()
                target = row['target'].lower()

                source_is_sensor = any(kw in source for kw in sensor_keywords)
                target_is_sensor = any(kw in target for kw in sensor_keywords)
                target_is_failure = 'failure' in target
                target_is_error = 'error' in target
                source_is_error = 'error' in source

                if source_is_sensor and target_is_failure:
                    cat = 'Sensor → Failure'
                elif source_is_sensor and target_is_error:
                    cat = 'Sensor → Error'
                elif source_is_error and target_is_failure:
                    cat = 'Error → Failure'
                elif source_is_sensor and target_is_sensor:
                    cat = 'Sensor → Sensor'
                else:
                    cat = 'Other'

                categories[cat].append({
                    'algorithm': algorithm,
                    'stability': row['stability']
                })

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        positions = []
        labels = []
        data_to_plot = []

        for i, (cat, edges) in enumerate(categories.items()):
            if edges:
                stabilities = [e['stability'] for e in edges]
                data_to_plot.append(stabilities)
                positions.append(i)
                labels.append(f"{cat}\n(n={len(edges)})")

        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)

        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Stability (%)', fontsize=12)
        ax.set_title('Edge Stability by Relationship Type', fontsize=14, fontweight='bold')
        ax.axhline(80, color='green', linestyle='--', alpha=0.7, label='80% threshold')
        ax.legend()

        plt.tight_layout()

        if save:
            fig.savefig(self.figures_path / 'stability_by_type.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {self.figures_path / 'stability_by_type.png'}")

        return fig

    def generate_all_plots(self, include_dataset: bool = False) -> None:
        """Generate all visualization plots.

        Args:
            include_dataset: Whether to also generate dataset figures
        """
        print("\nGenerating algorithm result visualizations...")

        self.plot_stability_distribution()
        self.plot_algorithm_agreement_heatmap()
        self.plot_top_edges_comparison()
        self.plot_consensus_graph()
        self.plot_edge_stability_by_type()

        print(f"\nAlgorithm figures saved to: {self.figures_path}")

        if include_dataset:
            self.generate_dataset_figures()

    def generate_dataset_figures(self) -> None:
        """Generate dataset overview figures."""
        print("\nGenerating dataset figures...")

        # Import and run dataset figure generation
        import sys
        from pathlib import Path

        # Add project root to path
        project_root = self.results_path.parent

        # Check if raw data exists
        raw_data_path = project_root / 'data' / 'raw'
        if not raw_data_path.exists() or not list(raw_data_path.glob('*.csv')):
            print("  Skipping dataset figures: raw data not found in data/raw/")
            print("  Download from: https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance")
            return

        try:
            # Try to import the dataset figure generator
            dataset_figures_script = project_root / 'generate_dataset_figures.py'
            if dataset_figures_script.exists():
                import subprocess
                result = subprocess.run(
                    [sys.executable, str(dataset_figures_script)],
                    cwd=str(project_root),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("  Dataset figures generated successfully")
                else:
                    print(f"  Dataset figure generation failed: {result.stderr}")
            else:
                print("  generate_dataset_figures.py not found")
        except Exception as e:
            print(f"  Error generating dataset figures: {e}")


@click.command()
@click.option('--results-path', default='results', help='Path to results directory')
@click.option('--min-stability', default=70.0, help='Minimum stability for consensus graph')
@click.option('--min-algorithms', default=3, help='Minimum algorithms for consensus graph')
@click.option('--include-dataset', is_flag=True, help='Also generate dataset overview figures')
def main(results_path: str, min_stability: float, min_algorithms: int, include_dataset: bool):
    """
    Generate visualizations for causal discovery results.
    """
    plotter = ResultsPlotter(results_path)
    plotter.generate_all_plots(include_dataset=include_dataset)


if __name__ == '__main__':
    main()
