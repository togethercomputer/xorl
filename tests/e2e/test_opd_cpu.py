"""CPU e2e-style validation for OPD server data path."""

import asyncio

import pytest
import torch

from tests._helpers.opd import make_teacher_files, reference_grouped_opd_loss
from xorl.distillation import TeacherActivationCache, TeacherHeadManager
from xorl.ops.loss import opd_loss_function
from xorl.server.backend import DummyBackend
from xorl.server.orchestrator.request_processor import RequestProcessor
from xorl.server.protocol.api_orchestrator import OrchestratorRequest, OutputType
from xorl.server.protocol.operations import ModelPassData
from xorl.server.runner.utils.batch_utils import convert_batch_to_tensors


pytestmark = [pytest.mark.e2e, pytest.mark.cpu, pytest.mark.server]


class OPDCPUBackend(DummyBackend):
    def __init__(self, student_hidden_table: torch.Tensor, student_head: torch.Tensor):
        super().__init__()
        self.student_hidden_table = student_hidden_table
        self.student_head = student_head
        self.last_batches = None
        self.last_routed_experts = None
        self.last_routed_expert_logits = None

    async def forward_backward(
        self,
        batches,
        loss_fn="causallm_loss",
        loss_fn_params=None,
        model_id=None,
        routed_experts=None,
        routed_expert_logits=None,
        request_id=None,
    ):
        assert loss_fn == "opd_loss"
        assert model_id == "opd-e2e"
        self.last_batches = batches
        self.last_routed_experts = routed_experts
        self.last_routed_expert_logits = routed_expert_logits

        params = loss_fn_params or {}
        head_manager = TeacherHeadManager(params["teacher_heads"])
        hidden_cache = TeacherActivationCache(params["teacher_hidden_caches"])

        total_loss = torch.tensor(0.0)
        valid_tokens = 0
        for raw_batch in batches:
            batch = convert_batch_to_tensors(raw_batch)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            teacher_ids = batch["teacher_ids"]
            local_valid = int((labels != -100).sum().item())
            valid_tokens += local_valid

            hidden_states = self.student_hidden_table[input_ids]
            local_loss = hidden_states.sum() * 0.0 + self.student_head.sum() * 0.0
            for teacher_id in torch.unique(teacher_ids[labels != -100]).tolist():
                teacher_id = int(teacher_id)
                teacher_labels = labels.masked_fill(teacher_ids != teacher_id, -100)
                teacher_hidden_states = hidden_cache.get(
                    teacher_id,
                    batch["teacher_cache_indices"],
                    device="cpu",
                    dtype=hidden_states.dtype,
                )
                teacher_head = head_manager.get(teacher_id, device="cpu", dtype=self.student_head.dtype)
                result = opd_loss_function(
                    hidden_states=hidden_states,
                    weight=self.student_head,
                    labels=teacher_labels,
                    teacher_hidden_states=teacher_hidden_states,
                    teacher_lm_head_weight=teacher_head,
                    teacher_weights=batch["teacher_weights"],
                    num_chunks=2,
                    normalization_denominator=torch.tensor(local_valid),
                )
                local_loss = local_loss + result.loss
            total_loss = total_loss + local_loss

        return {
            "total_loss": float(total_loss.item()),
            "global_valid_tokens": valid_tokens,
            "opd_kl": 0.123,
            "opd_num_teachers:max": 2,
            "execution_time": 0.0,
        }


def test_opd_request_processor_to_backend_e2e(tmp_path):
    torch.manual_seed(123)
    vocab_size = 17
    hidden_size = 6
    teacher_cache_size = 16

    student_hidden_table = torch.randn(vocab_size, hidden_size) / hidden_size**0.5
    student_head = torch.randn(vocab_size, hidden_size) / hidden_size**0.5
    teacher_heads = {
        "0": torch.randn(vocab_size, hidden_size) / hidden_size**0.5,
        "1": torch.randn(vocab_size, hidden_size) / hidden_size**0.5,
    }
    teacher_hidden_caches = {
        "0": torch.randn(teacher_cache_size, hidden_size) / hidden_size**0.5,
        "1": torch.randn(teacher_cache_size, hidden_size) / hidden_size**0.5,
    }

    teacher_files = make_teacher_files(tmp_path, teacher_heads, teacher_hidden_caches)

    backend = OPDCPUBackend(student_hidden_table=student_hidden_table, student_head=student_head)
    processor = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=16,
        enable_packing=True,
        pad_to_multiple_of=1,
        cp_size=1,
    )

    # Deliberately unsorted by teacher. RequestProcessor should group by teacher
    # before packing so a training mini-batch only loads one head at a time.
    data = [
        {
            "input_ids": [8, 9],
            "target_tokens": [9, 10],
            "teacher_id": 1,
            "teacher_weight": 0.5,
            "teacher_cache_indices": [4, 5],
        },
        {
            "input_ids": [1, 2, 3],
            "target_tokens": [2, 3, 4],
            "teacher_id": 0,
            "teacher_weight": 1.0,
            "teacher_cache_indices": [0, 1, 2],
        },
        {
            "input_ids": [11],
            "target_tokens": [12],
            "teacher_id": 1,
            "teacher_weight": 1.5,
            "teacher_cache_indices": [6],
        },
    ]
    request = OrchestratorRequest(
        operation="forward_backward",
        payload=ModelPassData(
            data=data,
            loss_fn="opd_loss",
            loss_fn_params={
                "teacher_heads": teacher_files.heads,
                "teacher_hidden_caches": teacher_files.hidden_caches,
                "opd_sort_by_teacher": True,
            },
            model_id="opd-e2e",
            routed_experts=["r1-a", "r0", "r1-b"],
            routed_expert_logits=["l1-a", "l0", "l1-b"],
        ),
    )

    output = asyncio.run(processor.execute_forward_backward(request))

    assert output.output_type == OutputType.FORWARD_BACKWARD
    assert output.finished is True
    result = output.outputs[0]
    assert result["success"] is True
    assert result["valid_tokens"] == 6
    assert result["opd_kl"] == 0.123
    assert result["opd_num_teachers:max"] == 2
    assert "is_opd_kl" not in result

    assert backend.last_routed_experts == ["r0", "r1-a", "r1-b"]
    assert backend.last_routed_expert_logits == ["l0", "l1-a", "l1-b"]
    assert backend.last_batches[0]["teacher_ids"] == [[0, 0, 0, 1, 1, 1]]

    expected = reference_grouped_opd_loss(
        backend.last_batches[0],
        student_hidden_table,
        student_head,
        teacher_hidden_caches,
        teacher_heads,
    )
    assert result["loss"] == pytest.approx(expected.item(), rel=1e-5, abs=1e-6)


class TeacherCacheCPUBackend(DummyBackend):
    async def forward(
        self,
        batches,
        loss_fn="causallm_loss",
        loss_fn_params=None,
        model_id=None,
        routed_experts=None,
        routed_expert_logits=None,
        request_id=None,
    ):
        assert loss_fn == "teacher_hidden_cache"
        assert model_id == "teacher"
        return {
            "total_loss": 0.0,
            "global_valid_tokens": 5,
            "teacher_hidden_cache": {
                "path": "/tmp/teacher_hidden.safetensors",
                "tensor_key": "hidden_states",
                "num_tokens": 5,
                "hidden_size": 6,
                "cache_indices_by_sample": [[0, 1, 2], [3, 4]],
            },
            "teacher_prefill_tokens": 5,
            "teacher_prefill_forward_compute_s": 0.25,
            "teacher_hidden_cache_write_s": 0.01,
            "execution_time": 0.3,
        }


def test_teacher_hidden_cache_metadata_passes_through_request_processor():
    backend = TeacherCacheCPUBackend()
    processor = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=16,
        enable_packing=True,
        pad_to_multiple_of=1,
        cp_size=1,
    )
    request = OrchestratorRequest(
        operation="forward",
        payload=ModelPassData(
            data=[
                {"input_ids": [1, 2, 3], "target_tokens": [1, 2, 3]},
                {"input_ids": [4, 5], "target_tokens": [4, 5]},
            ],
            loss_fn="teacher_hidden_cache",
            loss_fn_params={"teacher_hidden_cache_path": "/tmp/teacher_hidden.safetensors"},
            model_id="teacher",
        ),
    )

    output = asyncio.run(processor.execute_forward(request))

    assert output.output_type == OutputType.FORWARD
    result = output.outputs[0]
    assert result["teacher_hidden_cache"]["cache_indices_by_sample"] == [[0, 1, 2], [3, 4]]
    assert result["teacher_prefill_tokens"] == 5
    assert result["teacher_prefill_forward_compute_s"] == 0.25
