# Qwen head-DX RMS-dot default-policy hardening - PASS

Date: 2026-07-09 UTC

Lane: `MK_QWEN_HEADDX_RMS_DOT_PARTIALS` default policy

Source package:

- Worktree: `/tmp/xorl-oss-qwen-hdxrmsdot-7abb355-codex`
- Commit: `42e5b1d` (`default qwen head-dx rmsdot partials`)
- Parent: `965a0c1`
- Shared source checkout: not edited
- Direct promotion gate: `results/operator-gap/qwen-hdxrmsdot-default-42e5b1d-positive.md`

Hardening scope:

- `test_ops.py` from `experiments/fused-training-megakernel`
- `test_model.py` from `experiments/fused-training-megakernel`
- GPU: local GPU
- Cache: `/tmp/torchext-qwen-hdxrmsdot-hardening-42e5b1d-20260709T2340Z`
- Caveat: these are generic megakernel regression tests. They do not instantiate exact qwen4b-l2; the direct exact-qwen promotion evidence is the default-policy graph replay/timing gate.

Artifacts:

- `test_ops.py` log: `results/qwen-hdxrmsdot-hardening-testops-localgpu-42e5b1d-20260709T2340Z.log`
- `test_ops.py` log sha256: `caffa0236108b18e946f53d2063199c514d686cc59818b67ddf1c28cbf24e22c`
- `test_ops.py` log size: `53` lines / `3605` bytes
- `test_model.py` log: `results/qwen-hdxrmsdot-hardening-testmodel-localgpu-42e5b1d-20260709T2340Z.log`
- `test_model.py` log sha256: `2a2c374769513097c90b965ba7946486c9c25a35321fd7b406c2e928f696845c`
- `test_model.py` log size: `80` lines / `3891` bytes

Results:

- `test_ops.py`: `ALL OP TESTS PASSED`, `JOB_DONE rc=0 2026-07-09T23:43:03Z`.
- `test_model.py`: `ALL MODEL TESTS PASSED`, `JOB_DONE rc=0 2026-07-09T23:45:39Z`.
- `test_model.py` worst gradient relative errors: `qn.3 0.0213` for nano H256/L4/S512, `kn.0 0.0184` for D128 ragged S192; both below the test threshold `0.03`.
- Training sanity passed: normal step loss `9.0496 -> 5.6400`, graphed replay loss `9.0496 -> 5.4326`.
- Graph replay checks passed inside `test_model.py`, including in-place input rewrite.

Concurrent stale lane:

- A pre-existing `965a0c1` model hardening wrapper also finished cleanly while this ran: wrapper log `results/qwen-hdxrmsdot-hardening-model-wrapper-localgpu-965a0c1-20260709T2340Z.log` sha256 `29374c83f9a3657380a7f5a6b2f8c96eead403178fda93facf37ba3fc82bee40`, `18` lines / `633` bytes.
- Its default-model log sha256 is `5c947e6ad1557444328ad9543937caf2938d2fa2a4a90da7553f4c9e63fb17b8`, `82` lines / `3986` bytes.
- Its candidate-model log sha256 is `4ac32998b3f2c1739ede8cfae18a3dcbc32ae7f247ca38301ca213c5c638644d`, `82` lines / `3984` bytes.
- Both stale-lane model runs ended `rc=0`; no hardening locks or hardening processes remain.

Verdict:

Generic megakernel hardening passed on top of the default-policy source package. This strengthens the promotion package but does not replace the exact qwen4b-l2 route/parity/timing/graph evidence.
