"""Moved to ``xorl.ops._vendored.flashqla`` (issue #78 phase 1).

This stub aliases the package so old-path imports keep resolving for one
deprecation cycle.  Deep submodule imports through the old path create a
duplicate module instance — switch to the new path.
"""

import importlib as _importlib
import sys as _sys


_sys.modules[__name__] = _importlib.import_module("xorl.ops._vendored.flashqla")
