from types import SimpleNamespace

from xorl.trainers import trainer as trainer_module
from xorl.trainers.trainer import Trainer


def test_bootstrap_passes_ep_intranode_to_parallel_state(monkeypatch, tmp_path):
    captured = {}

    class FakeDevice:
        def set_device(self, device):
            captured["device"] = device

    train_args = SimpleNamespace(
        local_rank=0,
        global_rank=0,
        world_size=16,
        seed=42,
        enable_full_determinism=False,
        output_dir=str(tmp_path),
        use_wandb=False,
        data_parallel_mode="fsdp2",
        ckpt_manager="dcp",
        global_batch_size=2,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        data_parallel_size=2,
        data_parallel_replicate_size=1,
        data_parallel_shard_size=2,
        tensor_parallel_size=1,
        expert_parallel_size=8,
        pipeline_parallel_size=1,
        ulysses_parallel_size=8,
        ringattn_parallel_size=1,
        lm_head_tensor_parallel_size=1,
        cp_fsdp_mode="all",
        ep_intranode=False,
        moe_recomputed=False,
        moe_global_load_balancing=False,
        ce_mode="eager",
        ce_num_chunks=8,
        fsdp_sharded_lm_head_loss=False,
        fsdp_sharded_lm_head_loss_num_chunks=8,
        softmax_auxiliary_loss=False,
        auxiliary_loss_multiplier=0.0,
    )
    args = SimpleNamespace(train=train_args)
    trainer = Trainer.__new__(Trainer)
    trainer.args = args
    trainer._startup_metrics = {}
    trainer._wandb_initialized = False
    trainer._log_host_inventory = lambda: None
    trainer._maybe_log_startup_metrics = lambda *_, **__: None

    monkeypatch.setattr(trainer_module, "asdict", lambda _: {"train": {}})
    monkeypatch.setattr(trainer_module, "get_torch_device", lambda: FakeDevice())
    monkeypatch.setattr(trainer_module, "get_device_type", lambda: "cuda")
    monkeypatch.setattr(trainer_module, "get_nccl_backend", lambda: "nccl")
    monkeypatch.setattr(trainer_module.dist, "init_process_group", lambda *_, **__: None)
    monkeypatch.setattr(trainer_module.helper, "set_seed", lambda *_, **__: None)
    monkeypatch.setattr(trainer_module.helper, "enable_third_party_logging", lambda: None)
    monkeypatch.setattr(trainer_module, "save_args", lambda *_, **__: None)
    monkeypatch.setattr(trainer_module, "build_checkpointer", lambda *_, **__: "checkpointer")
    monkeypatch.setattr(trainer_module, "get_cpu_world_group", lambda: None)
    monkeypatch.setattr(trainer_module, "get_parallel_state", lambda: SimpleNamespace(device_mesh=None, ep_size=8))

    def fake_init_parallel_state(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(trainer_module, "init_parallel_state", fake_init_parallel_state)

    trainer._bootstrap()

    assert captured["device"] == "cuda:0"
    assert captured["ep_intranode"] is False
