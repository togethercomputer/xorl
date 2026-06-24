"""GLM-5 sparse-MLA + DSA-indexer kernels.

Vendored from the radixark `miles` glm5 plugin
(`miles_plugins/models/glm5/ops/`). All four kernel files are tilelang-only
— the dispatch layer in ``xorl.models.transformers.glm5.sparse_mla`` and
``...glm5.indexer`` lazy-imports tilelang and falls back to the pure-torch
reference paths when tilelang isn't installed or the input is on CPU.

We carry the bwd kernels even though current usage is fwd-only because
miles' `SparseMLA` / `IndexerFunction` autograd wrappers reference them at
class-definition time; vendoring fwd alone would force a runtime import
error on `import miles_plugins.models.glm5.ops.sparse_mla`.
"""
