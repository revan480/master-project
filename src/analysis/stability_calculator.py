"""
Stability Calculator Module

Computes stability scores for causal edges across machines.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
import click


class StabilityCalculator:
    """
    Calculates stability scores for causal relationships across machines.

    Stability Score = (Number of machines showing edge) / (Total machines) × 100%
    """

    def __init__(self, results_path: str):
        """
        Initialize the StabilityCalculator.

        Args:
            results_path: Path to results directory containing causal graphs
        """
        self.results_path = Path(results_path)
        self.graphs_path = self.results_path / 'causal_graphs'
        self.output_path = self.results_path / 'stability_scores'
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Store loaded results
        self.machine_results: Dict[int, Dict] = {}
        self.algorithms: List[str] = []
        self.variables: List[str] = []

    def load_results(self) -> None:
        """Load all causal graph results from JSON files."""
        print("Loading causal graph results...")

        result_files = list(self.graphs_path.glob("machine_*.json"))

        if not result_files:
            raise FileNotFoundError(f"No result files found in {self.graphs_path}")

        for file_path in tqdm(result_files, desc="Loading results"):
            machine_id = int(file_path.stem.split('_')[1])

            with open(file_path, 'r') as f:
                results = json.load(f)

            self.machine_results[machine_id] = results

            # Track algorithms and variables
            if results.get('algorithms'):
                for alg_name, alg_result in results['algorithms'].items():
                    if alg_name not in self.algorithms:
                        self.algorithms.append(alg_name)

                    if alg_result.get('variables') and not self.variables:
                        self.variables = alg_result['variables']

        print(f"Loaded results for {len(self.machine_results)} machines")
        print(f"Algorithms: {', '.join(self.algorithms)}")
        print(f"Variables: {len(self.variables)}")

    def extract_edges(self, algorithm: str, threshold: float = 0.0) -> Dict[int, List[Tuple[str, str]]]:
        """
        Extract edges for a specific algorithm across all machines.

        Args:
            algorithm: Algorithm name
            threshold: Minimum edge weight to include

        Returns:
            Dictionary mapping machine_id to list of (source, target) edges
        """
        machine_edges = {}

        for machine_id, results in self.machine_results.items():
            if algorithm not in results.get('algorithms', {}):
                continue

            alg_result = results['algorithms'][algorithm]

            if not alg_result.get('success', False):
                continue

            edges = []
            for edge in alg_result.get('edges', []):
                source, target, weight = edge
                if abs(weight) > threshold:
                    edges.append((source, target))

            machine_edges[machine_id] = edges

        return machine_edges

    def calculate_edge_stability(self, algorithm: str,
                                  threshold: float = 0.0) -> pd.DataFrame:
        """
        Calculate stability scores for all edges found by an algorithm.

        Args:
            algorithm: Algorithm name
            threshold: Minimum edge weight to include

        Returns:
            DataFrame with columns: source, target, count, stability, machines
        """
        machine_edges = self.extract_edges(algorithm, threshold)
        total_machines = len(machine_edges)

        if total_machines == 0:
            return pd.DataFrame(columns=['source', 'target', 'count', 'stability', 'machines'])

        # Count edge occurrences
        edge_counts = defaultdict(lambda: {'count': 0, 'machines': []})

        for machine_id, edges in machine_edges.items():
            for edge in edges:
                key = (edge[0], edge[1])
                edge_counts[key]['count'] += 1
                edge_counts[key]['machines'].append(machine_id)

        # Create DataFrame
        rows = []
        for (source, target), data in edge_counts.items():
            stability = data['count'] / total_machines * 100
            rows.append({
                'source': source,
                'target': target,
                'count': data['count'],
                'total_machines': total_machines,
                'stability': stability,
                'machines': data['machines']
            })

        df = pd.DataFrame(rows)

        if not df.empty:
            df = df.sort_values('stability', ascending=False).reset_index(drop=True)

        return df

    def calculate_all_stability(self, threshold: float = 0.0) -> Dict[str, pd.DataFrame]:
        """
        Calculate stability scores for all algorithms.

        Args:
            threshold: Minimum edge weight to include

        Returns:
            Dictionary mapping algorithm name to stability DataFrame
        """
        results = {}

        print("\nCalculating stability scores...")
        for algorithm in tqdm(self.algorithms, desc="Processing algorithms"):
            stability_df = self.calculate_edge_stability(algorithm, threshold)
            results[algorithm] = stability_df

            # Save individual algorithm results
            output_file = self.output_path / f"stability_{algorithm.replace('+', 'plus')}.csv"
            stability_df.to_csv(output_file, index=False)

        return results

    def get_high_confidence_edges(self, min_stability: float = 80.0,
                                   min_algorithms: int = 2) -> pd.DataFrame:
        """
        Get edges that appear with high stability across multiple algorithms.

        Args:
            min_stability: Minimum stability score to include (percentage)
            min_algorithms: Minimum number of algorithms that must agree

        Returns:
            DataFrame with high-confidence edges
        """
        # Collect high-stability edges from each algorithm
        edge_algorithm_map = defaultdict(list)

        for algorithm in self.algorithms:
            stability_df = self.calculate_edge_stability(algorithm)

            high_stability = stability_df[stability_df['stability'] >= min_stability]

            for _, row in high_stability.iterrows():
                edge_key = (row['source'], row['target'])
                edge_algorithm_map[edge_key].append({
                    'algorithm': algorithm,
                    'stability': row['stability'],
                    'count': row['count']
                })

        # Filter by minimum algorithms
        rows = []
        for edge_key, algorithms_data in edge_algorithm_map.items():
            if len(algorithms_data) >= min_algorithms:
                avg_stability = np.mean([a['stability'] for a in algorithms_data])
                alg_names = [a['algorithm'] for a in algorithms_data]

                rows.append({
                    'source': edge_key[0],
                    'target': edge_key[1],
                    'num_algorithms': len(algorithms_data),
                    'algorithms': ', '.join(alg_names),
                    'avg_stability': avg_stability,
                    'max_stability': max(a['stability'] for a in algorithms_data),
                    'min_stability': min(a['stability'] for a in algorithms_data)
                })

        df = pd.DataFrame(rows)

        if not df.empty:
            df = df.sort_values(['num_algorithms', 'avg_stability'],
                               ascending=[False, False]).reset_index(drop=True)

        return df

    def calculate_algorithm_agreement(self) -> pd.DataFrame:
        """
        Calculate pairwise agreement between algorithms.

        Returns:
            DataFrame with algorithm agreement matrix
        """
        n_algs = len(self.algorithms)
        agreement_matrix = np.zeros((n_algs, n_algs))

        # Get edges for each algorithm
        algorithm_edges = {}
        for algorithm in self.algorithms:
            machine_edges = self.extract_edges(algorithm)
            # Create set of all unique edges across machines
            all_edges = set()
            for edges in machine_edges.values():
                all_edges.update(edges)
            algorithm_edges[algorithm] = all_edges

        # Calculate Jaccard similarity
        for i, alg1 in enumerate(self.algorithms):
            for j, alg2 in enumerate(self.algorithms):
                edges1 = algorithm_edges[alg1]
                edges2 = algorithm_edges[alg2]

                if len(edges1) == 0 and len(edges2) == 0:
                    agreement_matrix[i, j] = 1.0
                elif len(edges1) == 0 or len(edges2) == 0:
                    agreement_matrix[i, j] = 0.0
                else:
                    intersection = len(edges1 & edges2)
                    union = len(edges1 | edges2)
                    agreement_matrix[i, j] = intersection / union if union > 0 else 0.0

        df = pd.DataFrame(agreement_matrix,
                         index=self.algorithms,
                         columns=self.algorithms)

        return df

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.

        Returns:
            Dictionary with summary statistics
        """
        report = {
            'total_machines': len(self.machine_results),
            'algorithms': self.algorithms,
            'variables': self.variables,
            'algorithm_stats': {},
            'top_edges': {},
            'high_confidence_edges': None
        }

        # Per-algorithm statistics
        for algorithm in self.algorithms:
            stability_df = self.calculate_edge_stability(algorithm)

            if not stability_df.empty:
                report['algorithm_stats'][algorithm] = {
                    'total_edges': len(stability_df),
                    'high_stability_edges': len(stability_df[stability_df['stability'] >= 80]),
                    'medium_stability_edges': len(stability_df[(stability_df['stability'] >= 50) & (stability_df['stability'] < 80)]),
                    'low_stability_edges': len(stability_df[stability_df['stability'] < 50]),
                    'avg_stability': stability_df['stability'].mean(),
                    'max_stability': stability_df['stability'].max()
                }

                # Top 5 edges
                report['top_edges'][algorithm] = stability_df.head(5)[['source', 'target', 'stability']].to_dict('records')

        # High confidence edges across algorithms
        high_conf_df = self.get_high_confidence_edges(min_stability=70, min_algorithms=3)
        report['high_confidence_edges'] = high_conf_df.to_dict('records') if not high_conf_df.empty else []

        return report

    def save_all_results(self) -> None:
        """Save all stability analysis results."""
        print("\nSaving results...")

        # Calculate and save stability for each algorithm
        self.calculate_all_stability()

        # High confidence edges
        high_conf_df = self.get_high_confidence_edges()
        high_conf_df.to_csv(self.output_path / 'high_confidence_edges.csv', index=False)

        # Algorithm agreement
        agreement_df = self.calculate_algorithm_agreement()
        agreement_df.to_csv(self.output_path / 'algorithm_agreement.csv')

        # Summary report
        report = self.generate_summary_report()
        with open(self.output_path / 'summary_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Results saved to: {self.output_path}")

        # Print summary
        print("\n" + "="*60)
        print("STABILITY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total machines analyzed: {report['total_machines']}")
        print(f"Algorithms: {len(self.algorithms)}")
        print(f"\nHigh-confidence edges (>=70% stability, >=3 algorithms):")

        if report['high_confidence_edges']:
            for edge in report['high_confidence_edges'][:10]:
                print(f"  {edge['source']} -> {edge['target']}: "
                      f"{edge['avg_stability']:.1f}% ({edge['num_algorithms']} algorithms)")
        else:
            print("  No high-confidence edges found")


@click.command()
@click.option('--results-path', default='results', help='Path to results directory')
@click.option('--threshold', default=0.0, help='Minimum edge weight threshold')
@click.option('--min-stability', default=80.0, help='Minimum stability for high-confidence edges')
def main(results_path: str, threshold: float, min_stability: float):
    """
    Calculate stability scores for causal edges across machines.
    """
    calculator = StabilityCalculator(results_path)

    # Load results
    calculator.load_results()

    # Save all results
    calculator.save_all_results()


if __name__ == '__main__':
    main()
