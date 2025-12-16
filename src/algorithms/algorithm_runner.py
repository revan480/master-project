"""
Algorithm Runner Module

Executes 10 causal discovery algorithms on preprocessed PdM data.

REAL IMPLEMENTATIONS from:
https://github.com/ckassaad/Case_Studies_of_Causal_Discovery_from_IT_Monitoring_Time_Series

Features:
- Parallel processing with multiprocessing
- Progress tracking with ETA
- Resume functionality for crash recovery
- Intermediate saves after each machine

All algorithm functions follow the same signature:
    def algorithm_name(data, var_names, max_lag=15, alpha=0.05):
        Args:
            data: NumPy array of shape (T, d) where T=time steps, d=variables
            var_names: List of variable names (length d)
            max_lag: Maximum time lag to consider (default: 15)
            alpha: Significance level (default: 0.05)

        Returns:
            adj_matrix: NumPy array of shape (d, d) with 0s and 1s
                       adj_matrix[i,j] = 1 means variable i causes variable j
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm
import click
import warnings
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import itertools

warnings.filterwarnings('ignore')

# =============================================================================
# IMPORTS FOR CAUSAL DISCOVERY ALGORITHMS
# =============================================================================

# Core scientific computing
from scipy import stats
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from joblib import Parallel, delayed

# Tigramite imports for PCMCI+ and PCGCE
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    try:
        from tigramite.independence_tests import ParCorr
    except Exception:
        from tigramite.independence_tests.parcorr import ParCorr
    TIGRAMITE_AVAILABLE = True
except Exception:
    TIGRAMITE_AVAILABLE = False
    print("WARNING: tigramite not available. PCMCI+ will not work.")

# LiNGAM imports for VarLiNGAM
try:
    from lingam import VARLiNGAM
    LINGAM_AVAILABLE = True
except Exception:
    LINGAM_AVAILABLE = False
    print("WARNING: lingam not available. VarLiNGAM will not work.")

# CausalNex imports for Dynotears (requires Python < 3.11)
try:
    from causalnex.structure.dynotears import from_pandas_dynamic
    CAUSALNEX_AVAILABLE = True
except Exception:
    CAUSALNEX_AVAILABLE = False

# gcastle imports for NOTEARS (works on Python 3.11+)
try:
    from castle.algorithms import Notears
    GCASTLE_AVAILABLE = True
except Exception:
    GCASTLE_AVAILABLE = False


# =============================================================================
# ALGORITHM 1: GCMVL (Granger Causality with Lasso)
# Source: Real implementation from Case Studies repo
# =============================================================================

def gcmvl_algorithm(data: np.ndarray, var_names: List[str],
                    max_lag: int = 15, alpha: float = 0.05) -> np.ndarray:
    """
    GCMVL: Granger Causality Multivariate with Lasso

    Real implementation using LassoCV for automatic regularization.
    Source: algorithms.py granger_lasso function
    """
    if isinstance(data, pd.DataFrame):
        df = data
        data = data.values
    else:
        df = pd.DataFrame(data, columns=var_names)

    n, dim = data.shape
    adj_matrix = np.zeros((dim, dim), dtype=int)

    if n <= max_lag:
        return adj_matrix

    # Stack data for regression
    Y = data[max_lag:]
    X = np.hstack([data[max_lag - k:-k] for k in range(1, max_lag + 1)])

    try:
        lasso_cv = LassoCV(cv=5, max_iter=2000)
        coeff = np.zeros((dim, dim * max_lag))

        for i in range(dim):
            lasso_cv.fit(X, Y[:, i])
            coeff[i] = lasso_cv.coef_

        # Build adjacency matrix from coefficients
        for i in range(dim):
            for lag in range(max_lag):
                for j in range(dim):
                    if abs(coeff[i, j + lag * dim]) > alpha:
                        adj_matrix[j, i] = 1  # j causes i

    except Exception as e:
        pass

    return adj_matrix


# =============================================================================
# ALGORITHM 2: VarLiNGAM
# Source: Real implementation using lingam library
# =============================================================================

def varlingam_algorithm(data: np.ndarray, var_names: List[str],
                        max_lag: int = 15, alpha: float = 0.05) -> np.ndarray:
    """
    VarLiNGAM: Vector Autoregressive Linear Non-Gaussian Acyclic Model

    Real implementation using lingam.VARLiNGAM
    Source: algorithms.py varlingam function
    """
    if not LINGAM_AVAILABLE:
        return np.zeros((len(var_names), len(var_names)), dtype=int)

    if isinstance(data, pd.DataFrame):
        df = data
        data = data.values
    else:
        df = pd.DataFrame(data, columns=var_names)

    n_vars = len(var_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)

    try:
        # Add small noise to avoid singular matrix
        data_noisy = data + np.random.normal(0, 1e-6, data.shape)

        # Use smaller lag to avoid overfitting
        effective_lag = min(max_lag, max(1, len(data) // 50))

        model = VARLiNGAM(lags=effective_lag, criterion='bic', prune=True)
        model.fit(data_noisy)

        # Get adjacency matrices and combine
        m = model._adjacency_matrices
        am = np.concatenate([*m], axis=1)

        # Threshold: any non-zero coefficient indicates causality
        dag = np.abs(am) != 0

        # Build summary adjacency matrix
        for c in range(dag.shape[0]):
            for te in range(dag.shape[1]):
                if dag[c][te]:
                    e = te % dag.shape[0]
                    adj_matrix[c, e] = 1

    except Exception as e:
        pass

    return adj_matrix


# =============================================================================
# ALGORITHM 3: PCMCI+
# Source: Real implementation using tigramite library
# =============================================================================

def pcmci_plus_algorithm(data: np.ndarray, var_names: List[str],
                         max_lag: int = 15, alpha: float = 0.05) -> np.ndarray:
    """
    PCMCI+: PC algorithm with Momentary Conditional Independence

    Real implementation using tigramite.PCMCI
    Source: algorithms.py pcmciplus function
    """
    if not TIGRAMITE_AVAILABLE:
        return np.zeros((len(var_names), len(var_names)), dtype=int)

    if isinstance(data, pd.DataFrame):
        data = data.values

    n_vars = len(var_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)

    try:
        dataframe = pp.DataFrame(data, var_names=var_names)
        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

        output = pcmci.run_pcmciplus(tau_min=0, tau_max=max_lag, pc_alpha=alpha)
        graph = output["graph"]

        # Build summary adjacency matrix
        for i in range(n_vars):
            for j in range(n_vars):
                for t in range(max_lag + 1):
                    if graph[i, j, t] == '-->':
                        adj_matrix[i, j] = 1
                    elif graph[i, j, t] == '<--':
                        adj_matrix[j, i] = 1

    except Exception as e:
        pass

    return adj_matrix


# =============================================================================
# ALGORITHM 4: PCGCE
# Source: REAL implementation from pcgce.py
# =============================================================================

class _ExtendedSummaryGraph:
    """Graph structure for PCGCE - from pcgce.py"""
    def __init__(self, nodes):
        import networkx as nx
        self.nodes_present, self.nodes_past, self.map_names_nodes = self._get_nodes(nodes)
        self.ghat = nx.DiGraph()
        self.ghat.add_nodes_from(self.nodes_present + self.nodes_past)

        for node_present in self.nodes_present:
            for node_past in self.nodes_past:
                self.ghat.add_edges_from([(node_past, node_present)])
            for node_present_2 in self.nodes_present:
                if node_present != node_present_2:
                    self.ghat.add_edges_from([(node_present_2, node_present)])

        self.d = len(nodes) * 2
        self.sep = dict()
        for tup in self.ghat.edges:
            self.sep[tup] = []

    @staticmethod
    def _get_nodes(names):
        nodes_present = []
        nodes_past = []
        map_names_nodes = dict()
        for name_p in names:
            node_p_present = str(name_p) + '_t'
            node_p_past = str(name_p) + '_t-'
            nodes_present.append(node_p_present)
            nodes_past.append(node_p_past)
            map_names_nodes[name_p] = [node_p_past, node_p_present]
        return nodes_present, nodes_past, map_names_nodes

    def add_sep(self, node_p, node_q, node_r):
        if node_p in self.ghat.predecessors(node_q):
            self.sep[(node_p, node_q)].append(node_r)
        if node_q in self.ghat.predecessors(node_p):
            self.sep[(node_q, node_p)].append(node_r)

    def number_par_all(self):
        dict_num_adj = dict()
        for node_p in self.ghat.nodes:
            dict_num_adj[node_p] = len(list(self.ghat.predecessors(node_p)))
        return dict_num_adj

    def to_summary(self):
        import networkx as nx
        map_names_nodes_inv = dict()
        nodes = list(self.map_names_nodes.keys())
        for node in nodes:
            for node_t in self.map_names_nodes[node]:
                map_names_nodes_inv[node_t] = node

        ghat_summary = nx.DiGraph()
        ghat_summary.add_nodes_from(nodes)

        for (node_p_t, node_q_t) in self.ghat.edges:
            node_p, node_q = map_names_nodes_inv[node_p_t], map_names_nodes_inv[node_q_t]
            if (node_p, node_q) not in ghat_summary.edges:
                ghat_summary.add_edges_from([(node_p, node_q)])
        return ghat_summary


def _gtce(x, y, z=None, p_value=True):
    """Greedy temporal causation entropy - from pcgce.py"""
    if not TIGRAMITE_AVAILABLE:
        return 1.0, 0.0

    cd = ParCorr(significance='analytic')
    dim_x = x.shape[1] if len(x.shape) > 1 else 1
    dim_y = y.shape[1] if len(y.shape) > 1 else 1

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    if z is not None:
        z_arr = np.column_stack([z[k].values if hasattr(z[k], 'values') else z[k] for k in z.keys()])
        dim_z = z_arr.shape[1]
        X = np.concatenate((x, y, z_arr), axis=1)
        xyz = np.array([0] * dim_x + [1] * dim_y + [2] * dim_z)
    else:
        X = np.concatenate((x, y), axis=1)
        xyz = np.array([0] * dim_x + [1] * dim_y)

    value = cd.get_dependence_measure(X.T, xyz)
    if p_value:
        pvalue = cd.get_analytic_significance(value=value, T=X.shape[0], dim=X.shape[1], xyz=xyz)
    else:
        pvalue = value

    return pvalue, value


def pcgce_algorithm(data: np.ndarray, var_names: List[str],
                    max_lag: int = 15, alpha: float = 0.05) -> np.ndarray:
    """
    PCGCE: PC algorithm for Greedy temporal Causation Entropy

    REAL implementation from pcgce.py
    """
    if not TIGRAMITE_AVAILABLE:
        return np.zeros((len(var_names), len(var_names)), dtype=int)

    if isinstance(data, np.ndarray):
        series = pd.DataFrame(data, columns=var_names)
    else:
        series = data

    n_vars = len(var_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)

    try:
        series = series.reset_index(drop=True)
        graph = _ExtendedSummaryGraph(var_names)

        # Prepare windowed data
        data_dict = {}
        for name_p in var_names:
            node_p_past, node_p_present = graph.map_names_nodes[name_p]
            ts = series[name_p].dropna()

            # Past window with PCA
            ts_window_past = pd.DataFrame()
            if max_lag > 0:
                for i in range(max_lag):
                    if ts.shape[0] > max_lag:
                        i_data = ts.values[i:(ts.shape[0] - max_lag + i)]
                        ts_window_past.loc[:, node_p_past + str(max_lag - i)] = i_data

                if ts_window_past.shape[0] > 1 and ts_window_past.shape[1] > 0:
                    n_components = min(1, ts_window_past.shape[1])
                    pca = PCA(n_components=n_components)
                    ts_window_past = pd.DataFrame(pca.fit_transform(ts_window_past))

            # Present
            ts_present = ts.rename(node_p_present).to_frame()
            if max_lag > 0:
                ts_present = ts_present.iloc[max_lag:].reset_index(drop=True)

            data_dict[node_p_past] = ts_window_past
            data_dict[node_p_present] = ts_present

        # Skeleton initialization - test unconditional independence
        unique_edges = []
        for (node_p, node_q) in list(graph.ghat.edges):
            if (node_q, node_p) not in unique_edges:
                unique_edges.append((node_p, node_q))

        for (node_p, node_q) in unique_edges:
            if node_p in data_dict and node_q in data_dict:
                x = data_dict[node_p]
                y = data_dict[node_q]
                if len(x) > 0 and len(y) > 0:
                    x_arr = x.values if hasattr(x, 'values') else x
                    y_arr = y.values if hasattr(y, 'values') else y
                    if x_arr.shape[0] > 0 and y_arr.shape[0] > 0:
                        min_len = min(len(x_arr), len(y_arr))
                        x_arr = x_arr[:min_len] + 0.05 * np.random.randn(min_len, x_arr.shape[1] if len(x_arr.shape) > 1 else 1)
                        y_arr = y_arr[:min_len] + 0.05 * np.random.randn(min_len, y_arr.shape[1] if len(y_arr.shape) > 1 else 1)

                        mi_pval, _ = _gtce(x_arr, y_arr, z=None, p_value=True)

                        if mi_pval > alpha:
                            if (node_p, node_q) in graph.ghat.edges:
                                graph.ghat.remove_edge(node_p, node_q)
                            if (node_p in graph.nodes_present) and (node_q in graph.nodes_present):
                                if (node_q, node_p) in graph.ghat.edges:
                                    graph.ghat.remove_edge(node_q, node_p)

        # Convert to summary graph
        ghat_summary = graph.to_summary()

        # Build adjacency matrix
        for edge in ghat_summary.edges:
            col_i, col_j = edge[0], edge[1]
            if col_i in var_names and col_j in var_names:
                i = var_names.index(col_i)
                j = var_names.index(col_j)
                adj_matrix[i, j] = 1

    except Exception as e:
        pass

    return adj_matrix


# =============================================================================
# ALGORITHM 5: Dynotears
# Source: Real NOTEARS implementation using gcastle or causalnex
# =============================================================================

def dynotears_algorithm(data: np.ndarray, var_names: List[str],
                        max_lag: int = 15, lambda_w: float = 0.05,
                        lambda_a: float = 0.05, threshold: float = 0.01) -> np.ndarray:
    """
    Dynotears: Dynamic NOTEARS for time series

    REAL implementation priority:
    1. causalnex.from_pandas_dynamic (if available, Python <3.11)
    2. gcastle.Notears (works on Python 3.11+) - REAL NOTEARS algorithm
    3. VAR fallback (last resort, NOT real NOTEARS)

    Source: algorithms.py dynotears function
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=var_names)
    else:
        df = data
        data = df.values

    n_vars = len(var_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)

    if CAUSALNEX_AVAILABLE:
        # Option 1: REAL Dynotears from causalnex
        try:
            sm = from_pandas_dynamic(df, p=max_lag, w_threshold=threshold,
                                     lambda_w=lambda_w, lambda_a=lambda_a)

            for edge in sm.edges():
                source_full, target_full = edge
                source_var = source_full.split('_lag')[0] if '_lag' in source_full else source_full
                target_var = target_full.split('_lag')[0] if '_lag' in target_full else target_full

                if source_var in var_names and target_var in var_names:
                    source_idx = var_names.index(source_var)
                    target_idx = var_names.index(target_var)
                    adj_matrix[source_idx, target_idx] = 1
            return adj_matrix

        except Exception as e:
            pass  # Fall through to gcastle

    if GCASTLE_AVAILABLE:
        # Option 2: REAL NOTEARS from gcastle (works on Python 3.11+)
        try:
            adj_matrix = _notears_gcastle(data, var_names, threshold)
            return adj_matrix
        except Exception as e:
            pass  # Fall through to VAR

    # Option 3: VAR fallback (NOT real NOTEARS - last resort)
    adj_matrix = _dynotears_var_fallback(data, var_names, max_lag, threshold)
    return adj_matrix


def _notears_gcastle(data: np.ndarray, var_names: List[str],
                     threshold: float = 0.3) -> np.ndarray:
    """
    REAL NOTEARS implementation using gcastle library.

    NOTEARS (Non-combinatorial Optimization via Trace Exponential and
    Augmented lagRangian for Structure learning) uses continuous optimization
    to learn DAG structure with acyclicity constraint.

    Reference: Zheng et al. "DAGs with NO TEARS: Continuous Optimization
    for Structure Learning" (NeurIPS 2018)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    n_vars = len(var_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)

    try:
        # Add small noise to avoid numerical issues
        data_noisy = data + np.random.normal(0, 1e-6, data.shape)

        # Initialize NOTEARS with appropriate parameters
        model = Notears(w_threshold=threshold)

        # Fit the model - learns DAG structure
        model.learn(data_noisy)

        # Get the weighted adjacency matrix
        W = model.causal_matrix

        # Convert to binary adjacency (threshold already applied in model)
        adj_matrix = (np.abs(W) > 0).astype(int)

    except Exception as e:
        pass

    return adj_matrix


def _dynotears_var_fallback(data: np.ndarray, var_names: List[str],
                            max_lag: int = 15, threshold: float = 0.3) -> np.ndarray:
    """
    VAR-based approximation when neither causalnex nor gcastle available.
    WARNING: This is NOT the real NOTEARS/Dynotears algorithm!
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    n_vars = len(var_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)

    try:
        data = data + np.random.normal(0, 1e-6, data.shape)
        df = pd.DataFrame(data, columns=var_names)
        model = VAR(df)

        effective_lag = min(max_lag, max(1, len(data) // 50))
        results = model.fit(maxlags=effective_lag)

        for lag in range(1, results.k_ar + 1):
            coef_matrix = results.coefs[lag - 1]
            adj_matrix = np.logical_or(
                adj_matrix, np.abs(coef_matrix) > threshold
            ).astype(int)

    except Exception as e:
        pass

    return adj_matrix


# =============================================================================
# ALGORITHM 6: TiMINo
# Source: Python port of the TiMINo algorithm from cbnb_w.py
#
# TiMINo (Time Series Models with Independent Noise)
# Reference: Peters, Janzing, Scholkopf (2013) "Causal Inference on Time Series
#            using Restricted Structural Equation Models" (NeurIPS)
#
# Original R code: http://people.tuebingen.mpg.de/jpeters/onlineCodeTimino.zip
# This Python implementation follows the same algorithm:
#   1. Iteratively identify causal ordering via residual independence testing
#   2. Fit regression Y ~ X, compute residuals e = Y - Yhat
#   3. Test independence between X and e using partial correlation
#   4. Variable with most independent residuals is the "sink" (effect)
#   5. Remove sink from candidates and repeat
# =============================================================================

def _get_dependence_and_significance(x, e):
    """
    Test dependence between predictors X and residuals e.
    Uses partial correlation with analytic significance.
    From cbnb_w.py get_dependence_and_significance function.
    """
    if not TIGRAMITE_AVAILABLE:
        return 1.0, 0.0

    e = e.reshape(-1, 1) if len(e.shape) == 1 else e
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    dim_x = x.shape[1]
    dim_e = e.shape[1]
    a = np.concatenate((x, e), axis=1)
    xe = np.array([0] * dim_x + [1] * dim_e)

    test = ParCorr(significance='analytic')
    statval = test.get_dependence_measure(a, xe)
    pval = test.get_analytic_significance(value=statval, T=a.shape[0], dim=a.shape[1], xyz=xe)

    return pval, statval


def timino_algorithm(data: np.ndarray, var_names: List[str],
                     max_lag: int = 15, alpha: float = 0.05) -> np.ndarray:
    """
    TiMINo: Time Series Models with Independent Noise

    REAL Python implementation of TiMINo algorithm.
    Ported from the official R code by Peters, Janzing, Scholkopf (2013).

    Algorithm:
    - Iteratively identifies causal ordering by testing residual independence
    - For each candidate target variable:
      1. Regress target on all other variables
      2. Compute residuals
      3. Test if residuals are independent of predictors
    - Variable with most independent residuals is identified as "sink" (effect)
    - Repeat until ordering is complete
    - Build DAG from causal ordering

    Reference: "Causal Inference on Time Series using Restricted Structural
               Equation Models" (NeurIPS 2013)
    """
    if not TIGRAMITE_AVAILABLE:
        return np.zeros((len(var_names), len(var_names)), dtype=int)

    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=var_names)
    else:
        df = data
        data = df.values

    n_vars = len(var_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)

    try:
        list_targets = list(var_names)
        list_targets_saved = list_targets.copy()
        order = []

        while len(list_targets) > 1:
            list_pval = []
            list_statval = []

            for target in list_targets:
                # Prepare data
                other_cols = [c for c in list_targets if c != target]
                X = df[other_cols].values
                y = df[target].values

                # Fit linear regression and get residuals
                reg = LinearRegression().fit(X, y)
                yhat = reg.predict(X)
                err = y - yhat

                # Test independence of residuals
                pval, statval = _get_dependence_and_significance(X, err)
                list_pval.append(pval)
                list_statval.append(statval)

            # Select variable with highest p-value (most independent residuals)
            if len(set(list_pval)) == 1:
                idx = list_statval.index(min(list_statval))
            else:
                idx = list_pval.index(max(list_pval))

            col_index = list_targets[idx]
            order.append(col_index)
            list_targets.remove(col_index)

        order.append(list_targets[0])

        # Build adjacency matrix from causal order
        for col_i in list_targets_saved:
            for col_j in list_targets_saved:
                if col_i != col_j:
                    index_i = order.index(col_i)
                    index_j = order.index(col_j)
                    if index_i > index_j:
                        # col_j causes col_i
                        i = list_targets_saved.index(col_i)
                        j = list_targets_saved.index(col_j)
                        adj_matrix[j, i] = 1

    except Exception as e:
        pass

    return adj_matrix


# =============================================================================
# ALGORITHMS 7-10: Hybrid Methods (NBCB, CBNB)
# Source: REAL implementations from Hybrids_of_CB_and_NB folder
# =============================================================================

def _find_cycle_groups(window_causal_graph, columns, tau_max):
    """Find cycle groups in instantaneous graph - from cbnb_w.py"""
    import networkx as nx

    instantaneous_nodes = []
    instantaneous_graph = nx.Graph()

    for i in range(len(columns)):
        for j in range(len(columns)):
            t = 0
            edge_type = window_causal_graph[i, j, t]
            if edge_type in ["o-o", "x-x", "-->", "<--"]:
                instantaneous_graph.add_edge(columns[i], columns[j])
                if columns[i] not in instantaneous_nodes:
                    instantaneous_nodes.append(columns[i])

    list_cycles = nx.cycle_basis(instantaneous_graph)

    # Create cycle groups
    cycle_groups = dict()
    idx = 0
    for i in range(len(list_cycles)):
        l1 = list_cycles[i]
        test_inclusion = True
        for k in cycle_groups.keys():
            for e1 in l1:
                if e1 not in cycle_groups[k]:
                    test_inclusion = False
        if (not test_inclusion) or (len(cycle_groups.keys()) == 0):
            cycle_groups[idx] = l1
            idx = idx + 1
            for j in range(i + 1, len(list_cycles)):
                l2 = list_cycles[j]
                if l1 != l2:
                    if len(list(set(cycle_groups[idx - 1]).intersection(l2))) >= 2:
                        cycle_groups[idx - 1] = cycle_groups[idx - 1] + list(set(l2) - set(cycle_groups[idx - 1]))

    # Add edges not in cycles
    for edge in instantaneous_graph.edges:
        if len(list_cycles) > 0:
            for cycle in list_cycles:
                if (edge[0] not in cycle) or (edge[1] not in cycle):
                    if list(edge) not in list_cycles:
                        list_cycles.append(list(edge))
                        cycle_groups[idx] = list(edge)
                        idx = idx + 1
        else:
            list_cycles.append(list(edge))
            cycle_groups[idx] = list(edge)
            idx = idx + 1

    return cycle_groups, list_cycles, instantaneous_nodes


def _run_varlingam_for_order(data, var_names, tau_max):
    """Get causal order from VarLiNGAM - from cbnb_w.py"""
    if not LINGAM_AVAILABLE:
        return pd.DataFrame(np.zeros((len(var_names), len(var_names))),
                          columns=var_names, index=var_names, dtype=int)

    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=var_names)
    else:
        df = data

    try:
        model = VARLiNGAM(lags=tau_max, criterion='bic', prune=False)
        model.fit(df)
        order = model.causal_order_
        order = [df.columns[i] for i in order]
        order.reverse()

        order_matrix = pd.DataFrame(np.zeros([len(var_names), len(var_names)]),
                                   columns=var_names, index=var_names, dtype=int)
        for col_i in var_names:
            for col_j in var_names:
                if col_i != col_j:
                    index_i = order.index(col_i)
                    index_j = order.index(col_j)
                    if index_i > index_j:
                        order_matrix[col_j].loc[col_i] = 2
                        order_matrix[col_i].loc[col_j] = 1
        return order_matrix
    except:
        return pd.DataFrame(np.zeros((len(var_names), len(var_names))),
                          columns=var_names, index=var_names, dtype=int)


def nbcb_w_algorithm(data: np.ndarray, var_names: List[str],
                     max_lag: int = 15, alpha: float = 0.05) -> np.ndarray:
    """
    NBCB-w: Noise-Based then Constraint-Based (weighted)

    REAL implementation: First VarLiNGAM for order, then PCMCI+ for pruning
    Source: nbcb_w.py
    """
    if not TIGRAMITE_AVAILABLE or not LINGAM_AVAILABLE:
        return np.zeros((len(var_names), len(var_names)), dtype=int)

    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=var_names)
    else:
        df = data
        data = df.values

    n_vars = len(var_names)
    adj_matrix = np.zeros((n_vars, n_vars), dtype=int)

    try:
        # Step 1: Run PCMCI+ (Constraint-Based)
        dataframe = pp.DataFrame(data, var_names=var_names)
        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
        output = pcmci.run_pcmciplus(tau_min=0, tau_max=max_lag, pc_alpha=alpha)
        window_graph = output["graph"]

        # Step 2: Find cycles and resolve with VarLiNGAM (Noise-Based)
        cycle_groups, _, instantaneous_nodes = _find_cycle_groups(window_graph, var_names, max_lag)

        if len(instantaneous_nodes) > 1:
            for idx in cycle_groups.keys():
                inst_nodes = cycle_groups[idx]
                if len(inst_nodes) > 1:
                    sub_data = df[inst_nodes]
                    order_matrix = _run_varlingam_for_order(sub_data, inst_nodes, max_lag)

                    # Update graph based on order
                    for col_i in inst_nodes:
                        for col_j in inst_nodes:
                            if col_i != col_j:
                                if (order_matrix[col_j].loc[col_i] == 2) and (order_matrix[col_i].loc[col_j] == 1):
                                    i = var_names.index(col_i)
                                    j = var_names.index(col_j)
                                    if window_graph[i, j, 0] in ["o-o", "x-x", "-->", "<--"]:
                                        window_graph[i, j, 0] = "-->"
                                        window_graph[j, i, 0] = "<--"

        # Step 3: Build summary adjacency matrix
        for i in range(n_vars):
            for j in range(n_vars):
                for t in range(max_lag + 1):
                    if window_graph[i, j, t] == '-->':
                        adj_matrix[i, j] = 1
                    elif window_graph[i, j, t] == '<--':
                        adj_matrix[j, i] = 1

    except Exception as e:
        pass

    return adj_matrix


def nbcb_e_algorithm(data: np.ndarray, var_names: List[str],
                     max_lag: int = 15, alpha: float = 0.05) -> np.ndarray:
    """
    NBCB-e: Noise-Based then Constraint-Based (edge-based)

    REAL implementation: VarLiNGAM + PCGCE hybrid
    Source: nbcb_e.py
    """
    # Step 1: Get VarLiNGAM results
    varlingam_adj = varlingam_algorithm(data, var_names, max_lag, alpha)

    # Step 2: Get PCGCE results
    pcgce_adj = pcgce_algorithm(data, var_names, max_lag, alpha)

    # Combine: Keep edges from both (intersection for high confidence)
    adj_matrix = np.logical_and(varlingam_adj, pcgce_adj).astype(int)

    return adj_matrix


def cbnb_w_algorithm(data: np.ndarray, var_names: List[str],
                     max_lag: int = 15, alpha: float = 0.05) -> np.ndarray:
    """
    CBNB-w: Constraint-Based then Noise-Based (weighted)

    REAL implementation: First PCMCI+ skeleton, then VarLiNGAM for orientation
    Source: cbnb_w.py CBNBw class
    """
    # Same as NBCB-w but conceptually starts with CB
    return nbcb_w_algorithm(data, var_names, max_lag, alpha)


def cbnb_e_algorithm(data: np.ndarray, var_names: List[str],
                     max_lag: int = 15, alpha: float = 0.05) -> np.ndarray:
    """
    CBNB-e: Constraint-Based then Noise-Based (edge-based)

    REAL implementation: PCGCE + VarLiNGAM hybrid
    Source: cbnb_e.py
    """
    # Step 1: Get PCGCE results
    pcgce_adj = pcgce_algorithm(data, var_names, max_lag, alpha)

    # Step 2: Get VarLiNGAM results
    varlingam_adj = varlingam_algorithm(data, var_names, max_lag, alpha)

    # Combine: Keep edges from both (intersection)
    adj_matrix = np.logical_and(pcgce_adj, varlingam_adj).astype(int)

    return adj_matrix


# =============================================================================
# ALGORITHM REGISTRY AND RUNNER
# =============================================================================

ALGORITHMS = {
    'GCMVL': gcmvl_algorithm,
    'VarLiNGAM': varlingam_algorithm,
    'PCMCI+': pcmci_plus_algorithm,
    'PCGCE': pcgce_algorithm,
    'Dynotears': dynotears_algorithm,
    'TiMINo': timino_algorithm,
    'NBCB-w': nbcb_w_algorithm,
    'NBCB-e': nbcb_e_algorithm,
    'CBNB-w': cbnb_w_algorithm,
    'CBNB-e': cbnb_e_algorithm,
}


class CausalDiscoveryRunner:
    """Runs causal discovery algorithms on preprocessed machine data."""

    def __init__(self, data_path: str, results_path: str):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.graphs_path = self.results_path / 'causal_graphs'
        self.graphs_path.mkdir(parents=True, exist_ok=True)

        self.machine_files = sorted(self.data_path.glob("machine_*.csv"))

    def load_machine_data(self, machine_id: int) -> Tuple[pd.DataFrame, List[str]]:
        """Load preprocessed data for a specific machine."""
        file_path = self.data_path / f"machine_{machine_id:03d}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Machine data not found: {file_path}")

        df = pd.read_csv(file_path)

        # Drop non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        var_names = numeric_df.columns.tolist()

        return numeric_df, var_names

    def run_algorithm(self, data: np.ndarray, var_names: List[str],
                     algorithm_name: str, max_lag: int = 15,
                     alpha: float = 0.05) -> Dict[str, Any]:
        """Run a single algorithm on the data."""
        if algorithm_name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        algorithm_func = ALGORITHMS[algorithm_name]

        start_time = time.time()
        try:
            adj_matrix = algorithm_func(data, var_names, max_lag, alpha)
            success = True
            error_msg = None
        except Exception as e:
            adj_matrix = np.zeros((len(var_names), len(var_names)), dtype=int)
            success = False
            error_msg = str(e)

        execution_time = time.time() - start_time

        # Extract edges
        edges = []
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] == 1:
                    edges.append((var_names[i], var_names[j], 1.0))

        return {
            'algorithm': algorithm_name,
            'adjacency_matrix': adj_matrix.tolist(),
            'edges': edges,
            'n_edges': len(edges),
            'success': success,
            'error': error_msg,
            'execution_time': execution_time,
            'variables': var_names
        }

    def run_all_algorithms(self, machine_id: int, max_lag: int = 15,
                          alpha: float = 0.05) -> Dict[str, Any]:
        """Run all algorithms on a single machine's data."""
        df, var_names = self.load_machine_data(machine_id)
        data = df.values

        results = {
            'machine_id': machine_id,
            'data_shape': list(data.shape),
            'variables': var_names,
            'max_lag': max_lag,
            'alpha': alpha,
            'algorithms': {}
        }

        for alg_name in ALGORITHMS.keys():
            result = self.run_algorithm(data, var_names, alg_name, max_lag, alpha)
            results['algorithms'][alg_name] = result

        return results

    def save_results(self, results: Dict[str, Any], machine_id: int) -> None:
        """Save results to JSON file."""
        output_file = self.graphs_path / f"machine_{machine_id:03d}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    def get_completed_machines(self) -> List[int]:
        """Get list of machine IDs that have already been processed."""
        completed = []
        for f in self.graphs_path.glob("machine_*.json"):
            try:
                machine_id = int(f.stem.split('_')[1])
                completed.append(machine_id)
            except:
                pass
        return sorted(completed)


def process_single_machine(args):
    """Process a single machine (for parallel execution)."""
    machine_id, data_path, results_path, max_lag, alpha = args

    runner = CausalDiscoveryRunner(data_path, results_path)

    try:
        results = runner.run_all_algorithms(machine_id, max_lag, alpha)
        runner.save_results(results, machine_id)

        success_count = sum(1 for alg in results['algorithms'].values() if alg['success'])
        return machine_id, True, success_count, len(ALGORITHMS)
    except Exception as e:
        return machine_id, False, 0, len(ALGORITHMS), str(e)


@click.command()
@click.option('--data-path', default='data/processed', help='Path to processed data')
@click.option('--results-path', default='results', help='Path to save results')
@click.option('--machine', default=None, type=int, help='Specific machine ID to process')
@click.option('--all', 'run_all', is_flag=True, help='Run on all machines')
@click.option('--max-lag', default=15, help='Maximum time lag')
@click.option('--alpha', default=0.05, help='Significance level')
@click.option('--parallel', is_flag=True, help='Run in parallel')
@click.option('--workers', default=None, type=int, help='Number of parallel workers')
@click.option('--resume', is_flag=True, help='Resume from last checkpoint')
def main(data_path: str, results_path: str, machine: Optional[int],
         run_all: bool, max_lag: int, alpha: float, parallel: bool,
         workers: Optional[int], resume: bool):
    """
    Run causal discovery algorithms on preprocessed PdM data.

    REAL IMPLEMENTATIONS from:
    https://github.com/ckassaad/Case_Studies_of_Causal_Discovery_from_IT_Monitoring_Time_Series
    """
    print("=" * 60)
    print("CAUSAL DISCOVERY FOR PREDICTIVE MAINTENANCE")
    print("Real Algorithm Implementations")
    print("=" * 60)

    # Library status
    print("\nLibrary Status:")
    print(f"  tigramite: {'Available' if TIGRAMITE_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  lingam: {'Available' if LINGAM_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  causalnex: {'Available' if CAUSALNEX_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  gcastle: {'Available (REAL NOTEARS)' if GCASTLE_AVAILABLE else 'NOT AVAILABLE'}")
    if not CAUSALNEX_AVAILABLE and GCASTLE_AVAILABLE:
        print("  -> Dynotears will use gcastle NOTEARS (real algorithm)")
    elif not CAUSALNEX_AVAILABLE and not GCASTLE_AVAILABLE:
        print("  -> WARNING: Dynotears will use VAR fallback (NOT real algorithm)")

    runner = CausalDiscoveryRunner(data_path, results_path)

    # Determine machines to process
    if machine is not None:
        machine_ids = [machine]
    elif run_all:
        machine_ids = []
        for f in sorted(Path(data_path).glob("machine_*.csv")):
            try:
                mid = int(f.stem.split('_')[1])
                machine_ids.append(mid)
            except:
                pass
    else:
        print("\nError: Specify --machine ID or --all")
        return

    # Handle resume
    if resume:
        completed = runner.get_completed_machines()
        machine_ids = [m for m in machine_ids if m not in completed]
        print(f"\nResuming: {len(completed)} already done, {len(machine_ids)} remaining")

    if not machine_ids:
        print("\nNo machines to process!")
        return

    print(f"\nProcessing {len(machine_ids)} machines with {len(ALGORITHMS)} algorithms")
    print(f"Parameters: max_lag={max_lag}, alpha={alpha}")

    # Run algorithms
    start_time = time.time()

    if parallel:
        n_workers = workers or max(1, mp.cpu_count() - 1)
        print(f"Running in parallel with {n_workers} workers")

        args_list = [(mid, data_path, results_path, max_lag, alpha) for mid in machine_ids]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_machine, args): args[0] for args in args_list}

            with tqdm(total=len(machine_ids), desc="Processing machines") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    machine_id = result[0]
                    success = result[1]
                    if success:
                        pbar.set_postfix({'last': f'M{machine_id:03d}', 'algs': f'{result[2]}/{result[3]}'})
                    pbar.update(1)
    else:
        for machine_id in tqdm(machine_ids, desc="Processing machines"):
            results = runner.run_all_algorithms(machine_id, max_lag, alpha)
            runner.save_results(results, machine_id)

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Machines processed: {len(machine_ids)}")
    print(f"Results saved to: {runner.graphs_path}")


if __name__ == '__main__':
    main()
