# Qwen head-DX RMS-dot partials default-policy gate - POSITIVE

Date: 2026-07-09 UTC

Lane: `MK_QWEN_HEADDX_RMS_DOT_PARTIALS` / `_hdxrmsdot`

Source package:

- Worktree: `/tmp/xorl-oss-qwen-hdxrmsdot-7abb355-codex`
- Commit: `42e5b1d` (`default qwen head-dx rmsdot partials`)
- Parent: `965a0c1` (`add qwen head-dx rmsdot partials gate`)
- Base: `7abb355`
- Shared source checkout: not edited
- Source scope: `experiments/fused-training-megakernel/model.py`
- Change: exact qwen4b-l2 requests RMS-dot partials by default, while `MK_QWEN_HEADDX_RMS_DOT_PARTIALS=0` remains the rollback guard and existing PDF head-dX + H2560 RMS-dX dependency checks still gate the compiled route.
- Replay patch: `results/qwen-hdxrmsdot-default-42e5b1d-20260709T2336Z.patch`
- Patch sha256: `1576000007ca672b578d847ea9f774a3d5e7a61285710f1068171844b434cb9e`
- Patch size: `35` lines / `2007` bytes
- Apply-check: patch applies cleanly to detached `965a0c1` in `/tmp/xorl-oss-hdxrmsdot-default-applycheck-965a0c1-42e5b1d`
- Static checks: `py_compile` and scoped `git diff --check` passed

Helper:

- Path: `results/qwen_hdxrmsdot_default_policy_42e5b1d.py`
- sha256: `8ef1dd96a3c32d2e699b56b6d0a7836e3dd4276a157464b068d50f3f632fd28a`
- Size: `294` lines / `11228` bytes
- Static checks: `py_compile`, `--help`, and scoped `git diff --check` passed

Authoritative artifacts:

- Log: `results/qwen-hdxrmsdot-default-localgpu-42e5b1d-20260709T2339Z.log`
- Log sha256: `de275d159de6a52de9c74708ecd093554e1db7695326d095673a7b9dc6dee9ee`
- Log size: `16` lines / `39586` bytes
- Summary: `results/qwen-hdxrmsdot-default-summary-42e5b1d-20260709T2339Z.json`
- Summary sha256: `64e739889e5347c1261f1df6d8bd7a245f73fa11d557dc8e0507a85713a119fd`
- Summary size: `816` lines / `22895` bytes
- JSON verdict: `pass=true`, `diagnostic_complete=true`, `timing_positive=true`
- Job: `JOB_DONE rc=0 2026-07-09T23:39:33Z`
- GPU: local GPU; lock released and local GPU observed `0 MiB, 0%` after close
- local GPU: untouched

Default-policy gates:

- `policy` build used `MK_QWEN_HEADDX_RMS_DOT_PARTIALS` unset and selected `_hdxrmsdot`.
- `rollback` build used `MK_QWEN_HEADDX_RMS_DOT_PARTIALS=0` and selected the no-`_hdxrmsdot` route.
- Env rollback proof passed after every build in both orders.
- Route gate passed in both `rollback_first` and `policy_first`.
- Graph capture succeeded for policy and rollback in both orders.
- Graph-vs-explicit equivalence passed for policy and rollback in both orders.
- Graph parity between policy and rollback passed in both orders.
- Route stayed `78/44/14`, `smem=151552`, sidecar boundary active.
- Policy head-DX row `40` writes partials arg `80`, `nparts=10`, `X=2`, `wf=72`.
- Policy first final RMS dX row `42` consumes partials arg `80`, `nparts=10`; later H2560 RMS rows do not consume partials.

Graph replay timing:

- `rollback_first`: rollback median `9182.512283325195us`, policy median `9171.728134155273us`, median delta `-10.784149169921875us`, paired mean delta `-7.222612698872884us`, policy wins `31/48`.
- `policy_first`: rollback median `9245.02420425415us`, policy median `9208.048343658447us`, median delta `-36.975860595703125us`, paired mean delta `-44.43993171056112us`, policy wins `35/48`.

Numerical checks:

- `rollback_first` graph parity: loss diff `-3.814697265625e-06`, worst grad `emb`, abs `0.000244140625`, rel `0.009615384615384616`.
- `policy_first` graph parity: loss diff `0.0`, worst grad `emb`, abs `0.000244140625`, rel `0.009615384615384616`.
- Thresholds: `loss_atol=0.005`, `grad_rtol=0.05`, `grad_atol=0.0005`.

Verdict:

The exact qwen4b-l2 `_hdxrmsdot` route is promotion-ready as the no-env default policy. The rollback guard `MK_QWEN_HEADDX_RMS_DOT_PARTIALS=0` remains validated and returns the previous route.
