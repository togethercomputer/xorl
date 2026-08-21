"""Serving-parity twins vendored from / mirrored into SGLang (issue #78).

These modules exist in both engines so trainer and serving arithmetic share
one implementation:

- ``bi_families_v2`` — vendored byte-identical into the serving engine and
  sha256-gated; it keeps that engine's formatting (black, 88 columns) and is
  excluded from all rewriting hooks.  Any edit requires the paired
  serving-side edit.
- ``batch_invariant_ops`` — vendored-adapted from SGLang's
  ``srt/batch_invariant_ops`` (SGLang-internal helpers stubbed, DeepGEMM
  routing added).  Kept as a single file so it stays diffable against the
  serving twin; edits must consider the serving side.

Unlike ``ops/_vendored/`` (untouchable third-party snapshots), these are
first-party-maintained with a paired-edit discipline.
"""
