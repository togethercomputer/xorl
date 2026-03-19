# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported from torchforge tests.

import torch


def assert_close(actual, expected, atol=1e-4, rtol=1e-4):
    """Assert two tensors are close within tolerance."""
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def get_metric(metrics: dict, key: str):
    """Get a metric value from the metrics dict."""
    if key not in metrics:
        raise KeyError(f"Metric '{key}' not found. Available: {list(metrics.keys())}")
    return metrics[key]
