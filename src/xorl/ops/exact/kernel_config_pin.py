"""First-class kernel/toolchain pinning for exact contracts.

Byte-exact programs are TOOLCHAIN-SCOPED claims: different Triton or FA4
builds can compile the same source into different arithmetic. Autotune configs
are likewise per-process unless pinned: Triton's ``cache_results`` replays
tuned configs from the cache directory, so all ranks and the qualification
oracle must share one seeded cache. Each rank receives a separate clone to
avoid concurrent writes to the shared seed.

The admission contract is mechanical:

- A qualification run SEEDS a pin directory: its triton cache plus a
  toolchain manifest (torch/triton/flash-attn versions).
- Admission (``pin_exact_kernel_configs``) FAILS CLOSED unless
  ``XORL_EXACT_KERNEL_CONFIG_DIR`` names a seeded pin directory whose
  manifest matches the running toolchain, then points this process's
  ``TRITON_CACHE_DIR`` at a per-rank clone of the seeded cache. It must run
  BEFORE the first kernel compilation (admission time satisfies this).
"""

from __future__ import annotations

import json
import logging
import os
import shutil

import torch
import triton


logger = logging.getLogger("xorl.kernel_config_pin")

PIN_DIR_ENV = "XORL_EXACT_KERNEL_CONFIG_DIR"
MANIFEST_NAME = "toolchain_manifest.json"
CACHE_SUBDIR = "triton-cache"


class KernelConfigPinError(RuntimeError):
    """The kernel/toolchain pin is missing or violated. Fail closed."""


# Ownership sentinel: this module only ever deletes directories it created
# itself (the marker IS the authorization). Env-var-fed paths never reach
# rmtree without it.
OWNED_SENTINEL = ".xorl-kernel-pin-owned"


def _mark_owned(path: str) -> None:
    with open(os.path.join(path, OWNED_SENTINEL), "w") as f:
        f.write("created by xorl.ops.exact.kernel_config_pin; safe for it to replace\n")


def _rmtree_owned(path: str, *, containing_dir: str) -> None:
    """Delete ``path`` only if this module created it and it stays inside
    ``containing_dir`` after symlink resolution.

    The sentinel alone is not sufficient authorization: whoever controls the
    pin-directory env var could pre-place it anywhere.  Requiring the resolved
    target to live under the (also resolved) pin directory, and refusing
    symlinked targets, bounds any deletion to the pin tree itself.
    """
    if os.path.islink(path):
        raise KernelConfigPinError(f"refusing to delete {path!r}: it is a symlink")
    if not os.path.isdir(path):
        return
    resolved = os.path.realpath(path)
    container = os.path.realpath(containing_dir)
    if os.path.commonpath([resolved, container]) != container or resolved == container:
        raise KernelConfigPinError(
            f"refusing to delete {resolved!r}: it escapes the pin directory {container!r}",
        )
    if not os.path.isfile(os.path.join(path, OWNED_SENTINEL)):
        raise KernelConfigPinError(
            f"refusing to delete {path!r}: it lacks the ownership sentinel "
            f"{OWNED_SENTINEL!r} and was not created by this module. Remove or "
            "relocate it manually if it is stale.",
        )
    shutil.rmtree(path)


def _runtime_fingerprint() -> dict:
    # Use the distribution version rather than only the module attribute:
    # distinct flash-attn wheel builds can share a torch/triton fingerprint,
    # while some builds do not expose a useful ``flash_attn.__version__``.
    from importlib import metadata  # noqa: PLC0415

    fa = "unavailable"
    for dist in ("flash-attn-4", "flash_attn_4", "flash-attn", "flash_attn"):
        try:
            fa = metadata.version(dist)
            break
        except metadata.PackageNotFoundError:
            continue
    if fa == "unavailable":
        try:
            import flash_attn  # noqa: PLC0415

            fa = getattr(flash_attn, "__version__", "unavailable")
        except Exception:  # pragma: no cover - build dependent
            pass
    return {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "flash_attn": fa,
        "cuda": torch.version.cuda or "none",
    }


def seed_exact_kernel_config_pin(pin_dir: str, *, source_cache: str | None = None) -> dict:
    """Create/refresh a pin directory from the CURRENT runtime.

    Called by qualification runs (e.g. the fixture oracle phase) after their
    kernels have been tuned. Copies `source_cache` (default: the active
    TRITON_CACHE_DIR or ~/.triton/cache) into the pin and writes the
    toolchain manifest.
    """
    fingerprint = _runtime_fingerprint()
    pin_dir = os.path.realpath(pin_dir)
    parent = os.path.dirname(pin_dir)
    if not os.path.isdir(parent):
        raise KernelConfigPinError(
            f"pin directory parent {parent!r} does not exist; refusing to create a pin at an implausible location",
        )
    os.makedirs(pin_dir, exist_ok=True)
    cache_src = os.path.realpath(
        source_cache or os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    )
    cache_dst = os.path.join(pin_dir, CACHE_SUBDIR)
    if os.path.commonpath([cache_src, cache_dst]) == cache_src:
        raise KernelConfigPinError(
            f"seed source cache {cache_src!r} contains the pin destination {cache_dst!r}; "
            "copying would recurse into itself",
        )
    _rmtree_owned(cache_dst, containing_dir=pin_dir)
    if os.path.isdir(cache_src):
        shutil.copytree(cache_src, cache_dst)
    else:
        os.makedirs(cache_dst, exist_ok=True)
    _mark_owned(cache_dst)
    with open(os.path.join(pin_dir, MANIFEST_NAME), "w") as f:
        json.dump(fingerprint, f, indent=2)
    logger.info("exact kernel-config pin seeded at %s: %s", pin_dir, fingerprint)
    return fingerprint


def pin_exact_kernel_configs(*, rank: int | None = None) -> str:
    """Admission-time pin. Returns the per-rank TRITON_CACHE_DIR it installed.

    Fail-closed on: env unset, pin dir or manifest missing, toolchain
    fingerprint mismatch. Engagement-logged once per process.
    """
    pin_dir = os.environ.get(PIN_DIR_ENV)
    if not pin_dir:
        raise KernelConfigPinError(
            f"{PIN_DIR_ENV} is not set. Exact hybrid-Ulysses admission requires a seeded "
            "kernel/toolchain pin directory (seed_exact_kernel_config_pin from the "
            "qualification run). Byte claims are toolchain-scoped; refusing to run unpinned.",
        )
    # The env var feeds filesystem mutations below: resolve it and refuse
    # anything that is not an existing, seeded pin directory.
    pin_dir = os.path.realpath(pin_dir)
    if not os.path.isdir(pin_dir):
        raise KernelConfigPinError(
            f"{PIN_DIR_ENV}={pin_dir!r} is not an existing directory. Fail closed.",
        )
    manifest_path = os.path.join(pin_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise KernelConfigPinError(
            f"{PIN_DIR_ENV}={pin_dir!r} has no {MANIFEST_NAME}; the pin directory was never "
            "seeded by a qualification run. Fail closed.",
        )
    with open(manifest_path) as f:
        pinned = json.load(f)
    running = _runtime_fingerprint()
    mismatches = {k: (pinned.get(k), running[k]) for k in running if pinned.get(k) != running[k]}
    if mismatches:
        raise KernelConfigPinError(
            "Toolchain fingerprint mismatch against the kernel-config pin — the byte "
            f"qualification does not transfer: {mismatches}. Re-qualify or restore the "
            "pinned environment.",
        )
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    clone = os.path.join(pin_dir, "clones", f"rank{rank}")
    _rmtree_owned(clone, containing_dir=pin_dir)
    shutil.copytree(os.path.join(pin_dir, CACHE_SUBDIR), clone)
    _mark_owned(clone)
    os.environ["TRITON_CACHE_DIR"] = clone
    logger.info(
        "exact kernel-config pin engaged: %s (rank %d clone %s, toolchain %s)",
        pin_dir,
        rank,
        clone,
        running,
    )
    return clone
