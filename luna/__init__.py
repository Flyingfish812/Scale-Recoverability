"""
Luna — Scale-Resolved Field Reconstruction Evaluation Framework
==============================================================

Common layer: core data structures, wavelet operations, POD decomposition, model
definitions, and configuration management. It contains no executable script logic,
only reusable functions and classes.

Layer architecture:
    luna/core       — base types and constants
    luna/data       — data IO, dataset registry, mask generation
    luna/wavelet    — wavelet transform, band decomposition, error metrics
    luna/pod        — POD decomposition, projection, band-wise POD
    luna/models     — neural network model definitions (VCNN, Ridge, MLP)
    luna/config     — configuration schema and loaders
"""

__version__ = "2.0.0"
