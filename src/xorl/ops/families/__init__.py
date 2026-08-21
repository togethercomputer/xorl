"""Model-family-specific kernels (issue #78 phase 4).

One home per family. These stay under ``ops/`` rather than inside
``models/transformers/<family>/`` deliberately: the model packages have
import side effects (e.g. DeepSeek-V4 registers itself with the HF Auto
registries on import), and kernels must be importable without them.
"""
