import re

import torch.nn as nn


def set_module_from_path(model: nn.Module, path: str, value: any):
    attrs = path.split(".")
    if len(attrs) == 1:
        setattr(model, attrs[0], value)
    else:
        next_obj = getattr(model, attrs[0])
        set_module_from_path(next_obj, ".".join(attrs[1:]), value)


def get_module_from_path(model: nn.Module, path: str):
    attrs = path.split(".")
    if len(attrs) == 1:
        return getattr(model, attrs[0])
    else:
        next_obj = getattr(model, attrs[0])
        return get_module_from_path(next_obj, ".".join(attrs[1:]))


def check_fqn_match(fqn_pattern: str, fqn: str, prefix: str = None):
    assert isinstance(fqn_pattern, str), f"fqn_pattern must be a str, got {type(fqn_pattern)}"
    assert isinstance(fqn, str), f"fqn must be a str, got {type(fqn)}"

    if prefix:
        fqn_pattern = [".".join([prefix, pattern]) for pattern in fqn_pattern]

    regex_str = re.escape(fqn_pattern).replace(r"\*", r".*")
    regex_str = f"^{regex_str}$"
    regex = re.compile(regex_str)

    match = regex.match(fqn)

    return match
