"""DeepSeek V4 operator kernels and helpers.

The ``kernel`` submodule contains the TileLang-based kernels (sparse MLA fwd/bwd,
DSA indexer fwd/bwd, FP8 act-quant, hyper-connection sinkhorn). Higher-level
layer wrappers (compressor, indexer, attention) live alongside this package
in subsequent phases.
"""
