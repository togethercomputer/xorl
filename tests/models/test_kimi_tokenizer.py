import json

import pytest
from tiktoken.load import dump_tiktoken_bpe

from xorl.models import auto as auto_module
from xorl.models.auto import build_processor, build_tokenizer
from xorl.models.transformers.deepseek_v3.tokenization_kimi import TikTokenTokenizer


pytestmark = [pytest.mark.cpu]


def _write_tiktoken_fixture(tmp_path):
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
    return tokenizer_dir


def test_build_tokenizer_loads_local_kimi_tiktoken_without_remote_code(tmp_path):
    tokenizer_dir = _write_tiktoken_fixture(tmp_path)

    tokenizer = build_tokenizer(str(tokenizer_dir))

    assert isinstance(tokenizer, TikTokenTokenizer)
    assert tokenizer.bos_token_id == 256
    assert tokenizer.eos_token_id == 257
    assert tokenizer.pad_token_id == 259
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"


def test_build_tokenizer_fallback_does_not_enable_remote_code(monkeypatch, tmp_path):
    calls = {}

    def fake_from_pretrained(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(auto_module.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    build_tokenizer(str(tmp_path / "not-kimi"))

    assert "trust_remote_code" not in calls["kwargs"]
    assert calls["kwargs"]["padding_side"] == "right"


def test_build_processor_does_not_enable_remote_code(monkeypatch):
    calls = {}

    def fake_from_pretrained(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(auto_module.AutoProcessor, "from_pretrained", fake_from_pretrained)

    build_processor("processor-path")

    assert "trust_remote_code" not in calls["kwargs"]
    assert calls["kwargs"]["padding_side"] == "right"
