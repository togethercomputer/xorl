import json
from types import SimpleNamespace

import pytest
from tiktoken.load import dump_tiktoken_bpe

from xorl.models import auto as auto_module
from xorl.models.auto import _load_local_xorl_config, build_processor, build_tokenizer
from xorl.models.registry import get_registry
from xorl.models.transformers.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from xorl.models.transformers.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from xorl.models.transformers.deepseek_v3.tokenization_kimi import TikTokenTokenizer


pytestmark = [pytest.mark.cpu]


def _make_kimi_text_config():
    return dict(
        model_type="kimi_k2",
        architectures=["DeepseekV3ForCausalLM"],
        vocab_size=163840,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=8,
        n_shared_experts=2,
        n_group=4,
        topk_group=2,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        routed_scaling_factor=1.25,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        attention_bias=False,
        attention_dropout=0.0,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        rope_scaling={"rope_type": "default", "rope_theta": 1000000.0},
    )


def _assert_deepseek_v3_config_maps_kimi_wrapper_and_official_aux_defaults():
    hf_config = SimpleNamespace(
        model_type="kimi_k25",
        text_config=SimpleNamespace(**_make_kimi_text_config()),
        vision_config=SimpleNamespace(hidden_size=1024, num_hidden_layers=24),
        tie_word_embeddings=False,
    )

    config = DeepseekV3Config.from_hf_config(hf_config)

    assert config.model_type == "deepseek_v3"
    assert config.q_lora_rank == 32
    assert config.kv_lora_rank == 16
    assert config.qk_nope_head_dim == 16
    assert config.qk_rope_head_dim == 8
    assert config.v_head_dim == 16
    assert config.n_routed_experts == 8
    assert config.n_shared_experts == 2
    assert config.first_k_dense_replace == 1
    assert config.topk_method == "noaux_tc"
    assert config.scoring_func == "sigmoid"
    assert config.rope_scaling["rope_type"] == "default"
    assert config.rope_theta == 1000000.0

    # Official Kimi snapshots spell the auxiliary coefficient differently and
    # omit the derived router-output switch.
    kimi_text_config = _make_kimi_text_config()
    kimi_text_config.pop("router_aux_loss_coef")
    kimi_text_config.pop("output_router_logits")
    kimi_text_config["aux_loss_alpha"] = 0.001

    hf_config = SimpleNamespace(
        model_type="kimi_k25",
        text_config=SimpleNamespace(**kimi_text_config),
        tie_word_embeddings=False,
    )

    official_config = DeepseekV3Config.from_hf_config(hf_config)

    assert official_config.router_aux_loss_coef == pytest.approx(0.001)
    assert official_config.output_router_logits is True


def _assert_local_kimi_tokenizer_and_generic_fallback_loader_policy(monkeypatch, tmp_path):
    tokenizer_dir = tmp_path / "kimi-tokenizer"
    tokenizer_dir.mkdir()
    dump_tiktoken_bpe({bytes([i]): i for i in range(256)}, str(tokenizer_dir / "tiktoken.model"))
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TikTokenTokenizer",
                "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", None]},
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "additional_special_tokens": ["<|im_end|>"],
                "added_tokens_decoder": {
                    "256": {"content": "[BOS]", "special": True},
                    "257": {"content": "[EOS]", "special": True},
                    "258": {"content": "[UNK]", "special": True},
                    "259": {"content": "[PAD]", "special": True},
                    "260": {"content": "<|im_end|>", "special": True},
                },
            }
        )
    )

    tokenizer = build_tokenizer(str(tokenizer_dir))

    assert isinstance(tokenizer, TikTokenTokenizer)
    assert tokenizer.bos_token_id == 256
    assert tokenizer.eos_token_id == 257
    assert tokenizer.pad_token_id == 259
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"

    calls = []

    def capture(kind):
        def fake_from_pretrained(path, **kwargs):
            calls.append((kind, path, kwargs))
            return object()

        return fake_from_pretrained

    monkeypatch.setattr(auto_module.AutoTokenizer, "from_pretrained", capture("tokenizer"))
    monkeypatch.setattr(auto_module.AutoProcessor, "from_pretrained", capture("processor"))

    build_tokenizer(str(tmp_path / "not-kimi"))
    build_processor("processor-path")

    assert [kind for kind, _path, _kwargs in calls] == ["tokenizer", "processor"]
    for _kind, _path, kwargs in calls:
        assert "trust_remote_code" not in kwargs
        assert kwargs["padding_side"] == "right"


def test_kimi_wrapper_config_registry_and_local_loading_policy(monkeypatch, tmp_path):
    _assert_deepseek_v3_config_maps_kimi_wrapper_and_official_aux_defaults()

    registry = get_registry()
    assert "DeepseekV3ForCausalLM" in registry.supported_models
    assert registry.get_model_cls_from_model_arch("DeepseekV3ForCausalLM") is DeepseekV3ForCausalLM

    config_dir = tmp_path / "kimi-k25"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k25",
                "text_config": _make_kimi_text_config(),
                "vision_config": {
                    "hidden_size": 1024,
                    "num_hidden_layers": 24,
                },
                "tie_word_embeddings": False,
            }
        )
    )

    config = _load_local_xorl_config(str(config_dir), {})

    assert isinstance(config, DeepseekV3Config)
    assert config.model_type == "deepseek_v3"
    assert config.n_routed_experts == 8
    assert config.n_shared_experts == 2

    _assert_local_kimi_tokenizer_and_generic_fallback_loader_policy(monkeypatch, tmp_path)
