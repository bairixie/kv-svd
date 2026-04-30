"""
SVD methods for KV-cache compression experiments.

The public entry point is `run_svd` in `svd_api.py`. Individual
`random_cholesky_v*.py` files are preserved as research iterations so the
development path described in `blog.md` remains inspectable.
"""

from .svd_api import SVDConfig, run_svd

__all__ = ["SVDConfig", "run_svd"]
