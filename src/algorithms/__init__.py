"""
Causal discovery algorithms module.

Implements 10 causal discovery algorithms:
- GCMVL: Granger Causality with Multiple Variables using Lasso
- VarLiNGAM: Vector Autoregressive Linear Non-Gaussian Acyclic Model
- PCMCI+: PC algorithm with Momentary Conditional Independence
- PCGCE: PC algorithm with Granger Causality Extension
- Dynotears: Dynamic NOTEARS for time series
- TiMINo: Time Series Models with Independent Noise
- NBCB-w: Noise-Based Causal Bootstrapping (window)
- NBCB-e: Noise-Based Causal Bootstrapping (edge)
- CBNB-w: Constraint-Based Noise Bootstrapping (window)
- CBNB-e: Constraint-Based Noise Bootstrapping (edge)
"""

from .algorithm_runner import CausalDiscoveryRunner, ALGORITHMS
