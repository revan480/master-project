"""
Predictive Maintenance Insights Extractor

Extracts non-trivial, maintenance-relevant causal relationships from
the causal discovery results.

Filters out:
- Self-relationships (X -> X)
- Same-sensor relationships (volt_mean -> volt_max, etc.)

Focuses on:
- Cross-sensor relationships (vibration -> pressure)
- Sensor -> Error relationships
- Error -> Failure relationships
- Sensor -> Failure relationships (direct predictors)
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
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


class PdMInsightsExtractor:
    """Extract predictive maintenance insights from causal discovery results."""

    def __init__(self, results_path: str = 'results'):
        self.results_path = Path(results_path)
        self.stability_path = self.results_path / 'stability_scores'
        self.figures_path = self.results_path / 'figures'
        self.figures_path.mkdir(parents=True, exist_ok=True)

        # Load all stability data
        self.stability_data = {}
        self.algorithms = []
        self._load_stability_data()

        # Define variable categories
        self.sensor_types = ['volt', 'rotate', 'pressure', 'vibration']
        self.stat_suffixes = ['_mean', '_min', '_max', '_std']

    def _load_stability_data(self):
        """Load stability scores for all algorithms."""
        print("Loading stability data...")

        for f in self.stability_path.glob("stability_*.csv"):
            alg_name = f.stem.replace('stability_', '').replace('plus', '+')
            self.stability_data[alg_name] = pd.read_csv(f)
            if alg_name not in self.algorithms:
                self.algorithms.append(alg_name)

        # Load high confidence edges
        hc_file = self.stability_path / 'high_confidence_edges.csv'
        if hc_file.exists():
            self.high_confidence = pd.read_csv(hc_file)
        else:
            self.high_confidence = pd.DataFrame()

        print(f"Loaded {len(self.algorithms)} algorithms")

    def _get_variable_base(self, var_name: str) -> str:
        """Extract base variable name (e.g., 'volt' from 'volt_mean')."""
        for sensor in self.sensor_types:
            if var_name.startswith(sensor):
                return sensor
        if var_name.startswith('error'):
            return 'error'
        if var_name.startswith('failure'):
            return 'failure'
        if var_name.startswith('maint'):
            return 'maintenance'
        if var_name.startswith('age'):
            return 'age'
        return var_name

    def _is_trivial_edge(self, source: str, target: str) -> bool:
        """
        Check if edge is trivial (should be filtered out).

        Trivial edges:
        - Self-loops (X -> X)
        - Same sensor different stats (volt_mean -> volt_max)
        """
        # Self-loop
        if source == target:
            return True

        # Same sensor type
        source_base = self._get_variable_base(source)
        target_base = self._get_variable_base(target)

        if source_base == target_base and source_base in self.sensor_types:
            return True

        return False

    def _categorize_edge(self, source: str, target: str) -> str:
        """Categorize edge by type for PdM analysis."""
        source_base = self._get_variable_base(source)
        target_base = self._get_variable_base(target)

        # Failure-related (most important)
        if target_base == 'failure':
            if source_base == 'error':
                return 'Error -> Failure'
            elif source_base in self.sensor_types:
                return 'Sensor -> Failure'
            elif source_base == 'maintenance':
                return 'Maintenance -> Failure'
            elif source_base == 'age':
                return 'Age -> Failure'
            else:
                return 'Other -> Failure'

        # Error-related
        if target_base == 'error':
            if source_base in self.sensor_types:
                return 'Sensor -> Error'
            elif source_base == 'error':
                return 'Error -> Error'
            else:
                return 'Other -> Error'

        # Cross-sensor
        if source_base in self.sensor_types and target_base in self.sensor_types:
            return 'Sensor -> Sensor (cross)'

        # Sensor from failure (reverse causality or feedback)
        if source_base == 'failure' and target_base in self.sensor_types:
            return 'Failure -> Sensor'

        if source_base == 'error' and target_base in self.sensor_types:
            return 'Error -> Sensor'

        return 'Other'

    def extract_nontrivial_edges(self, min_stability: float = 60.0,
                                  min_algorithms: int = 2) -> pd.DataFrame:
        """
        Extract non-trivial edges with stability >= threshold.

        Args:
            min_stability: Minimum stability score (%)
            min_algorithms: Minimum number of algorithms agreeing

        Returns:
            DataFrame with non-trivial edges
        """
        print(f"\nExtracting non-trivial edges (stability >= {min_stability}%, algorithms >= {min_algorithms})...")

        # Collect edges from all algorithms
        edge_data = defaultdict(lambda: {
            'stabilities': {},
            'algorithms': []
        })

        for alg, df in self.stability_data.items():
            for _, row in df.iterrows():
                source, target = row['source'], row['target']

                # Skip trivial edges
                if self._is_trivial_edge(source, target):
                    continue

                stability = row['stability']
                if stability >= min_stability:
                    key = (source, target)
                    edge_data[key]['stabilities'][alg] = stability
                    if alg not in edge_data[key]['algorithms']:
                        edge_data[key]['algorithms'].append(alg)

        # Filter by minimum algorithms
        rows = []
        for (source, target), data in edge_data.items():
            if len(data['algorithms']) >= min_algorithms:
                avg_stability = np.mean(list(data['stabilities'].values()))
                max_stability = max(data['stabilities'].values())
                min_stability_val = min(data['stabilities'].values())

                category = self._categorize_edge(source, target)

                rows.append({
                    'source': source,
                    'target': target,
                    'category': category,
                    'num_algorithms': len(data['algorithms']),
                    'algorithms': ', '.join(sorted(data['algorithms'])),
                    'avg_stability': avg_stability,
                    'max_stability': max_stability,
                    'min_stability': min_stability_val,
                    'source_type': self._get_variable_base(source),
                    'target_type': self._get_variable_base(target)
                })

        df = pd.DataFrame(rows)

        if not df.empty:
            # Sort by category importance, then stability
            category_order = {
                'Error -> Failure': 0,
                'Sensor -> Failure': 1,
                'Maintenance -> Failure': 2,
                'Age -> Failure': 3,
                'Sensor -> Error': 4,
                'Error -> Error': 5,
                'Sensor -> Sensor (cross)': 6,
                'Failure -> Sensor': 7,
                'Error -> Sensor': 8,
                'Other -> Failure': 9,
                'Other -> Error': 10,
                'Other': 11
            }
            df['category_rank'] = df['category'].map(category_order)
            df = df.sort_values(['category_rank', 'avg_stability'],
                               ascending=[True, False]).reset_index(drop=True)
            df = df.drop('category_rank', axis=1)

        print(f"Found {len(df)} non-trivial edges")
        return df

    def get_top_pdm_edges(self, n: int = 20) -> pd.DataFrame:
        """Get top N predictive maintenance relevant edges."""
        # Get edges with lower threshold first
        all_edges = self.extract_nontrivial_edges(min_stability=50.0, min_algorithms=1)

        if all_edges.empty:
            print("No non-trivial edges found!")
            return pd.DataFrame()

        # Prioritize PdM-relevant categories
        pdm_categories = [
            'Error -> Failure',
            'Sensor -> Failure',
            'Maintenance -> Failure',
            'Age -> Failure',
            'Sensor -> Error',
            'Sensor -> Sensor (cross)',
            'Other -> Failure'
        ]

        pdm_edges = all_edges[all_edges['category'].isin(pdm_categories)]

        # If not enough, also include other categories
        if len(pdm_edges) < n:
            other_edges = all_edges[~all_edges['category'].isin(pdm_categories)]
            pdm_edges = pd.concat([pdm_edges, other_edges]).head(n)

        return pdm_edges.head(n)

    def create_pdm_summary_table(self) -> pd.DataFrame:
        """Create summary table of top PdM-relevant edges."""
        print("\n" + "="*80)
        print("TOP 20 PREDICTIVE MAINTENANCE EDGES")
        print("(Non-trivial, Cross-variable Relationships)")
        print("="*80)

        top_edges = self.get_top_pdm_edges(20)

        if top_edges.empty:
            print("No edges found!")
            return top_edges

        # Display table
        print(f"\n{'Rank':<5} {'Source':<20} {'Target':<20} {'Category':<25} {'Stab%':<8} {'#Alg':<5}")
        print("-"*85)

        for i, (_, row) in enumerate(top_edges.iterrows(), 1):
            print(f"{i:<5} {row['source']:<20} {row['target']:<20} "
                  f"{row['category']:<25} {row['avg_stability']:<8.1f} {row['num_algorithms']:<5}")

        # Save to CSV
        output_file = self.stability_path / 'top_pdm_edges.csv'
        top_edges.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")

        return top_edges

    def analyze_by_category(self) -> Dict[str, pd.DataFrame]:
        """Analyze edges by category."""
        all_edges = self.extract_nontrivial_edges(min_stability=50.0, min_algorithms=1)

        print("\n" + "="*80)
        print("EDGES BY CATEGORY")
        print("="*80)

        categories = all_edges['category'].unique()
        results = {}

        for cat in sorted(categories):
            cat_edges = all_edges[all_edges['category'] == cat]
            results[cat] = cat_edges

            print(f"\n{cat}: {len(cat_edges)} edges")
            if len(cat_edges) > 0:
                print(f"  Avg stability: {cat_edges['avg_stability'].mean():.1f}%")
                print(f"  Top edge: {cat_edges.iloc[0]['source']} -> {cat_edges.iloc[0]['target']} "
                      f"({cat_edges.iloc[0]['avg_stability']:.1f}%)")

        return results

    def visualize_pdm_network(self):
        """Create focused PdM network visualization."""
        try:
            import networkx as nx
        except ImportError:
            print("networkx not available")
            return

        top_edges = self.get_top_pdm_edges(30)

        if top_edges.empty:
            print("No edges to visualize")
            return

        print("\nGenerating PdM network visualization...")

        # Create graph
        G = nx.DiGraph()

        # Define category colors
        category_colors = {
            'Error -> Failure': '#d62728',      # Red (most important)
            'Sensor -> Failure': '#ff7f0e',     # Orange
            'Maintenance -> Failure': '#2ca02c', # Green
            'Age -> Failure': '#9467bd',        # Purple
            'Sensor -> Error': '#1f77b4',       # Blue
            'Error -> Error': '#e377c2',        # Pink
            'Sensor -> Sensor (cross)': '#17becf', # Cyan
            'Failure -> Sensor': '#bcbd22',     # Yellow-green
            'Error -> Sensor': '#7f7f7f',       # Gray
            'Other -> Failure': '#8c564b',      # Brown
            'Other -> Error': '#aec7e8',        # Light blue
            'Other': '#c7c7c7'                  # Light gray
        }

        # Add edges
        for _, row in top_edges.iterrows():
            G.add_edge(row['source'], row['target'],
                      weight=row['avg_stability'],
                      category=row['category'],
                      num_algorithms=row['num_algorithms'])

        # Node colors by type
        node_type_colors = {
            'volt': '#e41a1c',
            'rotate': '#377eb8',
            'pressure': '#4daf4a',
            'vibration': '#984ea3',
            'error': '#ff7f00',
            'failure': '#000000',
            'maintenance': '#2ca02c',
            'age': '#a65628'
        }

        node_colors = []
        for node in G.nodes():
            base = self._get_variable_base(node)
            node_colors.append(node_type_colors.get(base, '#999999'))

        # Layout
        fig, ax = plt.subplots(figsize=(16, 12))

        # Use shell layout with failure nodes in center
        failure_nodes = [n for n in G.nodes() if 'failure' in n]
        error_nodes = [n for n in G.nodes() if 'error' in n]
        sensor_nodes = [n for n in G.nodes() if any(s in n for s in self.sensor_types)]
        other_nodes = [n for n in G.nodes() if n not in failure_nodes + error_nodes + sensor_nodes]

        shells = []
        if failure_nodes:
            shells.append(failure_nodes)
        if error_nodes:
            shells.append(error_nodes)
        if sensor_nodes:
            shells.append(sensor_nodes)
        if other_nodes:
            shells.append(other_nodes)

        if len(shells) > 1:
            pos = nx.shell_layout(G, shells)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=1000, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        # Draw edges with category colors
        for (u, v, data) in G.edges(data=True):
            color = category_colors.get(data['category'], '#999999')
            width = data['weight'] / 25  # Scale by stability
            alpha = 0.5 + (data['num_algorithms'] / 10) * 0.5  # More algorithms = more opaque

            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                  edge_color=color, width=width,
                                  alpha=alpha, arrows=True, arrowsize=15,
                                  connectionstyle="arc3,rad=0.1", ax=ax)

        # Legend for node types
        node_legend = [
            mpatches.Patch(color=node_type_colors['volt'], label='Voltage'),
            mpatches.Patch(color=node_type_colors['rotate'], label='Rotation'),
            mpatches.Patch(color=node_type_colors['pressure'], label='Pressure'),
            mpatches.Patch(color=node_type_colors['vibration'], label='Vibration'),
            mpatches.Patch(color=node_type_colors['error'], label='Error'),
            mpatches.Patch(color=node_type_colors['failure'], label='Failure'),
        ]

        # Legend for edge types
        edge_legend = [
            mpatches.Patch(color=category_colors['Error -> Failure'], label='Error -> Failure'),
            mpatches.Patch(color=category_colors['Sensor -> Failure'], label='Sensor -> Failure'),
            mpatches.Patch(color=category_colors['Sensor -> Error'], label='Sensor -> Error'),
            mpatches.Patch(color=category_colors['Sensor -> Sensor (cross)'], label='Cross-Sensor'),
        ]

        leg1 = ax.legend(handles=node_legend, loc='upper left', title='Node Type', fontsize=9)
        ax.add_artist(leg1)
        ax.legend(handles=edge_legend, loc='upper right', title='Edge Type', fontsize=9)

        ax.set_title('Predictive Maintenance Causal Network\n'
                    '(Non-trivial Cross-Variable Relationships, Top 30 Edges)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig_pdm_network.png')
        plt.savefig(self.figures_path / 'fig_pdm_network.pdf')
        plt.close()
        print(f"Saved: {self.figures_path / 'fig_pdm_network.png'}")

    def visualize_category_summary(self):
        """Create bar chart of edges by category."""
        all_edges = self.extract_nontrivial_edges(min_stability=50.0, min_algorithms=1)

        if all_edges.empty:
            return

        # Count by category
        category_counts = all_edges['category'].value_counts()

        # Average stability by category
        category_stability = all_edges.groupby('category')['avg_stability'].mean()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Count by category
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
        bars1 = axes[0].barh(range(len(category_counts)), category_counts.values, color=colors)
        axes[0].set_yticks(range(len(category_counts)))
        axes[0].set_yticklabels(category_counts.index)
        axes[0].set_xlabel('Number of Edges')
        axes[0].set_title('(a) Non-Trivial Edges by Category')
        axes[0].invert_yaxis()

        # Add value labels
        for bar, val in zip(bars1, category_counts.values):
            axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', fontsize=9)

        # Plot 2: Average stability by category
        sorted_cats = category_stability.sort_values(ascending=False)
        colors2 = plt.cm.RdYlGn(sorted_cats.values / 100)
        bars2 = axes[1].barh(range(len(sorted_cats)), sorted_cats.values, color=colors2)
        axes[1].set_yticks(range(len(sorted_cats)))
        axes[1].set_yticklabels(sorted_cats.index)
        axes[1].set_xlabel('Average Stability (%)')
        axes[1].set_title('(b) Average Stability by Category')
        axes[1].set_xlim(0, 100)
        axes[1].invert_yaxis()

        # Add value labels
        for bar, val in zip(bars2, sorted_cats.values):
            axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.figures_path / 'fig_pdm_category_summary.png')
        plt.savefig(self.figures_path / 'fig_pdm_category_summary.pdf')
        plt.close()
        print(f"Saved: {self.figures_path / 'fig_pdm_category_summary.png'}")

    def generate_project_insights(self):
        """Generate all PdM insights for project."""
        print("\n" + "="*80)
        print("PREDICTIVE MAINTENANCE INSIGHTS ANALYSIS")
        print("="*80)

        # 1. Create summary table
        top_edges = self.create_pdm_summary_table()

        # 2. Analyze by category
        self.analyze_by_category()

        # 3. Generate visualizations
        self.visualize_pdm_network()
        self.visualize_category_summary()

        # 4. Key findings summary
        print("\n" + "="*80)
        print("KEY FINDINGS FOR project")
        print("="*80)

        if not top_edges.empty:
            # Count edges by category
            cat_counts = top_edges['category'].value_counts()

            print("\nEdge Distribution in Top 20:")
            for cat, count in cat_counts.items():
                print(f"  {cat}: {count}")

            # Highlight failure-related edges
            failure_edges = top_edges[top_edges['target_type'] == 'failure']
            print(f"\nFailure-Related Edges: {len(failure_edges)}")

            if len(failure_edges) > 0:
                print("\nTop Failure Predictors:")
                for i, (_, row) in enumerate(failure_edges.head(5).iterrows(), 1):
                    print(f"  {i}. {row['source']} -> {row['target']} "
                          f"({row['avg_stability']:.1f}%, {row['num_algorithms']} algorithms)")

            # Highlight error-related edges
            error_edges = top_edges[top_edges['target_type'] == 'error']
            print(f"\nError-Related Edges: {len(error_edges)}")

            if len(error_edges) > 0:
                print("\nTop Error Predictors:")
                for i, (_, row) in enumerate(error_edges.head(5).iterrows(), 1):
                    print(f"  {i}. {row['source']} -> {row['target']} "
                          f"({row['avg_stability']:.1f}%, {row['num_algorithms']} algorithms)")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nFiles saved to: {self.figures_path}")
        print(f"  - fig_pdm_network.png/pdf")
        print(f"  - fig_pdm_category_summary.png/pdf")
        print(f"  - top_pdm_edges.csv (in stability_scores/)")


def main():
    """Run PdM insights extraction."""
    extractor = PdMInsightsExtractor(results_path='results')
    extractor.generate_project_insights()


if __name__ == '__main__':
    main()
