"""
custom_kv — Error-Corrected Ultra-Low Precision KV Cache for Llama-3.

Public API:
    ErrorCorrectedCache  — drop-in HuggingFace KV cache (pass as past_key_values)
    ecc_cache            — context manager (recommended for single-request use)
    patch_model          — manual patch for production serving
    unpatch_model        — restore original SDPA
    calibrate_from_model — compute per-layer Lloyd-Max + alpha config
"""
from .cache import ErrorCorrectedCache
from .context import ecc_cache
from .patch import patch_model, unpatch_model
from .calibration import calibrate_from_model, load_calibration_config

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "ErrorCorrectedCache",
    "ecc_cache",
    "patch_model",
    "unpatch_model",
    "calibrate_from_model",
    "load_calibration_config",
]
