# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported from torchforge tests for GRPOLoss.

import pytest
import torch

from tests.ops.loss.conftest import assert_close
from xorl.ops.loss import TokenPartial, drgrpo_loss_function


@pytest.fixture
def inputs():
    """Test inputs for DRGRPO loss.

    Note: xorl's drgrpo_loss_function takes hidden_states and weight instead of logits.
    We create hidden_states and weight such that hidden_states @ weight.T produces
    logits with similar scale to torch.randn(B, S, V).
    """
    torch.manual_seed(42)
    B, S, V = 2, 4, 10
    H = 16  # Hidden dimension

    # Create hidden_states and weight matrix
    # Scale by 1/sqrt(H) so that logits = hidden_states @ weight.T has variance ~1
    # (similar to torch.randn logits in torchforge tests)
    hidden_states = torch.randn(B, S, H) / (H**0.5)
    weight = torch.randn(V, H)

    # Compute effective logits for reference (should have variance ~1)
    logits = hidden_states @ weight.T

    # Target IDs (called 'labels' in xorl API)
    labels = torch.randint(0, V, (B, S))

    # Seq 0: mild divergence, Seq 1: high divergence (triggers clipping)
    old_logprobs = torch.tensor(
        [
            [-2.0, -2.1, -1.9, -2.0],
            [-6.0, -1.0, -5.0, -0.5],
        ]
    )
    ref_logprobs = torch.randn(B, S) * 0.5 - 2.0
    advantages = torch.randn(B, S)

    # Interleaved mask: use ignore_index to mark non-loss positions
    # loss_mask in torchforge: [[1, 0, 1, 0], [1, 1, 0, 0]]
    # For xorl, we use ignore_index=-100 in labels for masked positions
    ignore_index = -100
    labels_with_mask = labels.clone()
    mask_pattern = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    labels_with_mask[~mask_pattern] = ignore_index

    return {
        "B": B,
        "S": S,
        "V": V,
        "H": H,
        "hidden_states": hidden_states,
        "weight": weight,
        "logits": logits,
        "labels": labels,
        "labels_with_mask": labels_with_mask,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "advantages": advantages,
        "ignore_index": ignore_index,
        "mask_pattern": mask_pattern,
    }


class TestDRGRPOLoss:
    """Tests for drgrpo_loss_function.

    Note: test_forward and test_backward contain regression tests with exact expected
    values. If the implementation changes intentionally, update the expected values by
    running the tests with pytest -v and recording the actual values:

        pytest tests/ops/loss/test_drgrpo_loss.py -v -k "test_forward or test_backward"

    Then update the assert_close(...) calls with the new values.
    """

    def test_forward_backward_and_metrics(self, inputs):
        """Forward value, gradients, and metric schema form one numerical contract."""
        d = inputs
        hidden_states = d["hidden_states"].clone().requires_grad_(True)

        output = drgrpo_loss_function(
            hidden_states=hidden_states,
            weight=d["weight"],
            labels=d["labels_with_mask"],
            old_logprobs=d["old_logprobs"],
            advantages=d["advantages"],
            ref_logprobs=d["ref_logprobs"],
            ignore_index=d["ignore_index"],
            clip_low=0.2,
            clip_high=0.2,
            beta=0.1,
        )

        assert output.loss.isfinite()
        assert output.loss.shape == ()
        # Regression test: expected value computed with seed=42 fixture inputs.
        # Default loss_reducer is TokenPartial(scale=mask.sum()); fixture has 4
        # active tokens of 8 → 2× the previous numel-scaled value.
        assert_close(output.loss, torch.tensor(0.727356))
        expected_keys = {
            "loss/ratio/mean",
            "loss/kl_policy/mean",
            "loss/clip/clipped_ratio/mean",
            "loss/clip/high_fraction",
            "loss/clip/low_fraction",
            "loss/kl_ref/mean",
        }
        assert expected_keys <= output.metrics.keys()

        output.loss.backward()
        assert hidden_states.grad is not None
        assert hidden_states.grad.isfinite().all()
        assert_close(hidden_states.grad.norm(), torch.tensor(3.028616))

        self._assert_zero_loss_boundaries(inputs)
        self._assert_positive_advantages_encourage_high_prob(inputs)
        self._assert_kl_penalty_requires_reference_and_affects_loss(inputs)
        self._assert_logprob_temperature_changes_behavior_k3()

    def _assert_logprob_temperature_changes_behavior_k3(self):
        hidden_states = torch.tensor([[[1.0, -0.5], [0.25, 0.75]]])
        weight = torch.tensor([[0.5, -1.0], [-0.25, 0.75], [1.0, 0.5]])
        labels = torch.tensor([[0, 2]])
        advantages = torch.zeros_like(labels, dtype=torch.float32)
        raw_logprobs = torch.log_softmax(hidden_states @ weight.T, dim=-1)
        old_logprobs = raw_logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        raw = drgrpo_loss_function(
            hidden_states=hidden_states,
            weight=weight,
            labels=labels,
            old_logprobs=old_logprobs,
            advantages=advantages,
            beta=0.0,
            ce_mode="eager",
        )
        behavior = drgrpo_loss_function(
            hidden_states=hidden_states,
            weight=weight,
            labels=labels,
            old_logprobs=old_logprobs,
            advantages=advantages,
            beta=0.0,
            ce_mode="eager",
            logprob_temperature=0.7,
        )

        expected_behavior_logprobs = torch.log_softmax((hidden_states @ weight.T) / 0.7, dim=-1)
        expected_behavior_logprobs = expected_behavior_logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        torch.testing.assert_close(raw.per_token_logprobs, old_logprobs)
        torch.testing.assert_close(behavior.per_token_logprobs, expected_behavior_logprobs)
        torch.testing.assert_close(raw.metrics["loss/kl_policy/mean"], torch.tensor(0.0), atol=1e-6, rtol=0.0)
        assert behavior.metrics["loss/kl_policy/mean"].abs() > 1e-6

    def _assert_zero_loss_boundaries(self, inputs):
        """Zero advantages, no trainable labels, and empty sequences remain finite and zero."""
        d = inputs
        advantages = torch.zeros_like(d["advantages"])

        output = drgrpo_loss_function(
            hidden_states=d["hidden_states"],
            weight=d["weight"],
            labels=d["labels_with_mask"],
            old_logprobs=d["old_logprobs"],
            advantages=advantages,
            ignore_index=d["ignore_index"],
            beta=0.0,
        )

        assert output.loss.isfinite()
        assert output.loss.abs() < 1e-5

        all_ignored = torch.full_like(d["labels"], d["ignore_index"])
        output = drgrpo_loss_function(
            hidden_states=d["hidden_states"],
            weight=d["weight"],
            labels=all_ignored,
            old_logprobs=d["old_logprobs"],
            advantages=d["advantages"],
            ignore_index=d["ignore_index"],
            beta=0.0,
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0

        B, V, H = 2, 10, 16
        hidden_states = torch.empty(B, 0, H)
        weight = torch.randn(V, H)
        labels = torch.empty(B, 0, dtype=torch.long)
        advantages = torch.empty(B, 0)
        old_logprobs = torch.empty(B, 0)

        output = drgrpo_loss_function(
            hidden_states=hidden_states,
            weight=weight,
            labels=labels,
            old_logprobs=old_logprobs,
            advantages=advantages,
            beta=0.0,
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0

    def _assert_positive_advantages_encourage_high_prob(self, inputs):
        """With positive advantages, higher target probability yields lower loss."""
        d = inputs
        B, S, V, H = d["B"], d["S"], d["V"], d["H"]

        # Create two scenarios with same structure but different logit magnitudes
        torch.manual_seed(123)
        labels = torch.randint(0, V, (B, S))
        old_logprobs = torch.zeros(B, S)
        positive_advantages = torch.ones(B, S) * 2.0

        # High probability scenario: hidden states that produce high logits for targets
        hidden_high = torch.randn(B, S, H)
        weight_high = torch.randn(V, H)
        # Bias the logits toward target tokens
        for b in range(B):
            for s in range(S):
                weight_high[labels[b, s]] += hidden_high[b, s] * 5.0

        # Low probability scenario
        hidden_low = torch.randn(B, S, H)
        weight_low = torch.randn(V, H)
        # Bias away from target tokens
        for b in range(B):
            for s in range(S):
                weight_low[labels[b, s]] -= hidden_low[b, s] * 5.0

        loss_high = drgrpo_loss_function(
            hidden_states=hidden_high,
            weight=weight_high,
            labels=labels,
            old_logprobs=old_logprobs,
            advantages=positive_advantages,
            beta=0.0,
        )

        loss_low = drgrpo_loss_function(
            hidden_states=hidden_low,
            weight=weight_low,
            labels=labels,
            old_logprobs=old_logprobs,
            advantages=positive_advantages,
            beta=0.0,
        )

        # Higher probability should yield lower (more negative) loss
        assert loss_high.loss < loss_low.loss

    def _assert_kl_penalty_requires_reference_and_affects_loss(self, inputs):
        """Positive KL weight requires reference logprobs and changes the loss."""
        d = inputs

        with pytest.raises(ValueError, match="ref_logprobs required"):
            drgrpo_loss_function(
                hidden_states=d["hidden_states"],
                weight=d["weight"],
                labels=d["labels_with_mask"],
                old_logprobs=d["old_logprobs"],
                advantages=d["advantages"],
                ref_logprobs=None,
                ignore_index=d["ignore_index"],
                beta=0.1,
            )

        loss_no_kl = drgrpo_loss_function(
            hidden_states=d["hidden_states"],
            weight=d["weight"],
            labels=d["labels_with_mask"],
            old_logprobs=d["old_logprobs"],
            advantages=d["advantages"],
            ignore_index=d["ignore_index"],
            beta=0.0,
        )

        loss_with_kl = drgrpo_loss_function(
            hidden_states=d["hidden_states"],
            weight=d["weight"],
            labels=d["labels_with_mask"],
            old_logprobs=d["old_logprobs"],
            advantages=d["advantages"],
            ref_logprobs=d["ref_logprobs"],
            ignore_index=d["ignore_index"],
            beta=0.1,
        )

        assert loss_no_kl.loss != loss_with_kl.loss
        assert loss_with_kl.loss.isfinite()
        # KL metrics should be present when beta > 0
        assert "loss/kl_ref/mean" in loss_with_kl.metrics

    def test_microbatch_composition(self, inputs):
        """Per-mb partial shares sum to single-batch values for both loss and metrics.

        Regression test for the metric-inflation bug: with global-denominator
        reducers, summing per-mb outputs must recover the single-batch result —
        not N times as large.
        """
        d = inputs
        B = d["B"]
        assert B >= 2, "Test requires B >= 2 to form micro-batches."

        loss_mask = (d["labels_with_mask"] != d["ignore_index"]).float()
        metric_reducer = TokenPartial(scale=loss_mask.sum())
        loss_reducer = TokenPartial(scale=torch.tensor(float(loss_mask.numel())))

        def call(slc):
            return drgrpo_loss_function(
                hidden_states=d["hidden_states"][slc],
                weight=d["weight"],
                labels=d["labels_with_mask"][slc],
                old_logprobs=d["old_logprobs"][slc],
                advantages=d["advantages"][slc],
                ref_logprobs=d["ref_logprobs"][slc],
                ignore_index=d["ignore_index"],
                beta=0.1,
                loss_reducer=loss_reducer,
                metric_reducer=metric_reducer,
            )

        single = call(slice(None))
        mbs = [call(slice(b, b + 1)) for b in range(B)]

        assert_close(sum(mb.loss for mb in mbs), single.loss)
        for key, expected in single.metrics.items():
            assert_close(sum(mb.metrics[key] for mb in mbs), expected)
