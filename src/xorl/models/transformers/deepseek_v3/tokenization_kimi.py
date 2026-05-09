import os
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from shutil import copyfile
from typing import Dict, Iterator, List, Optional, Tuple, Union, cast

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tokenizers import AddedToken
from transformers.tokenization_utils import PreTrainedTokenizer


try:
    from transformers.convert_slow_tokenizer import bytes_to_unicode
except ImportError:  # pragma: no cover - compatibility with older Transformers
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


logger = getLogger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tiktoken.model"}


def _normalize_added_tokens_decoder(added_tokens_decoder: Optional[dict]) -> Optional[dict[int, AddedToken]]:
    if added_tokens_decoder is None:
        return None

    normalized = {}
    for token_id, token_config in added_tokens_decoder.items():
        token_id = int(token_id)
        if isinstance(token_config, AddedToken):
            normalized[token_id] = token_config
        elif isinstance(token_config, dict):
            normalized[token_id] = AddedToken(
                token_config["content"],
                single_word=token_config.get("single_word", False),
                lstrip=token_config.get("lstrip", False),
                rstrip=token_config.get("rstrip", False),
                normalized=token_config.get("normalized", False),
                special=token_config.get("special", False),
            )
        else:
            normalized[token_id] = AddedToken(str(token_config), special=True)
    return normalized


class TikTokenTokenizer(PreTrainedTokenizer):
    """Local Kimi TikToken tokenizer.

    Moonshot Kimi snapshots publish this tokenizer as remote HuggingFace code.
    Xorl vendors the small tokenizer implementation locally so loading pinned
    local snapshots does not require enabling HuggingFace remote code.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    num_reserved_special_tokens = 256

    special_tokens: Dict[str, int]

    pat_str = "|".join(
        [
            r"""[\p{Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

    def __init__(
        self,
        vocab_file: str,
        bos_token: Union[str, AddedToken] = "[BOS]",
        eos_token: Union[str, AddedToken] = "[EOS]",
        unk_token: Optional[Union[str, AddedToken]] = None,
        pad_token: Optional[Union[str, AddedToken]] = None,
        additional_special_tokens: Optional[List[str]] = None,
        added_tokens_decoder: Optional[dict] = None,
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(vocab_file)

        if additional_special_tokens is None:
            additional_special_tokens = [
                "<|im_end|>",
                "<|im_user|>",
                "<|im_assistant|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "[EOT]",
                "<|im_system|>",
                "<|im_middle|>",
            ]

        added_tokens_decoder = _normalize_added_tokens_decoder(added_tokens_decoder)
        special_tokens_mapping = {}
        if added_tokens_decoder:
            special_tokens_mapping = {token_id: token.content for token_id, token in added_tokens_decoder.items()}

        self.vocab_file = vocab_file
        mergeable_ranks = load_tiktoken_bpe(vocab_file)
        num_base_tokens = len(mergeable_ranks)
        self.special_tokens = {
            special_tokens_mapping.get(i, f"<|reserved_token_{i}|>"): i
            for i in range(num_base_tokens, num_base_tokens + self.num_reserved_special_tokens)
        }

        self.model = tiktoken.Encoding(
            name=Path(vocab_file).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Loaded local Kimi tiktoken model from {vocab_file}")

        self.n_words: int = self.model.n_vocab
        self.bos_id: int = self.special_tokens[str(bos_token)]
        self.eos_id: int = self.special_tokens[str(eos_token)]
        self.pad_id: Optional[int] = self.special_tokens[str(pad_token)] if pad_token is not None else None
        self.unk_id: Optional[int] = self.special_tokens[str(unk_token)] if unk_token is not None else None

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.decoder = {}
        special_token_ids = set(self.special_tokens.values())
        for i in range(self.n_words):
            if i in special_token_ids:
                continue
            decoding = "".join(
                self.byte_encoder[ord(char)] for char in self.model.decode_single_token_bytes(i).decode("latin-1")
            )
            self.decoder[i] = decoding
        self.decoder.update({token_id: token for token, token_id in self.special_tokens.items()})

        self.encoder = {token: token_id for token_id, token in self.decoder.items()}
        self._token_config_cache = OrderedDict()
        self._cache_max_size = 128

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            added_tokens_decoder=added_tokens_decoder,
            **kwargs,
        )
        self.all_special_ids_set = set(self.all_special_ids)

    def encode(self, text: str, allow_special_tokens: bool = True, **kwargs) -> List[int]:
        if len(kwargs) > 0:
            logger.warning(f"Calling super().encode with {kwargs}")
            return super().encode(text, **kwargs)

        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        tiktoken_max_encode_chars = 400_000
        max_no_whitespaces_chars = 25_000
        substrs = []
        for processed_text in self.pre_tokenizer_process(text):
            substrs.extend(
                substr
                for i in range(0, len(processed_text), tiktoken_max_encode_chars)
                for substr in self._split_whitespaces_or_nonwhitespaces(
                    processed_text[i : i + tiktoken_max_encode_chars],
                    max_no_whitespaces_chars,
                )
            )

        token_ids: List[int] = []
        for substr in substrs:
            if allow_special_tokens:
                token_ids.extend(self.model.encode(substr, allowed_special="all"))
            else:
                token_ids.extend(self.model.encode(substr, disallowed_special=()))
        return token_ids

    def decode(self, token_ids: Union[int, List[int]], **kwargs) -> str:
        if len(kwargs) > 0:
            return super().decode(token_ids, **kwargs)
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.model.decode(cast(List[int], token_ids))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i, char in enumerate(s):
            is_now_space = char.isspace()
            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

    def pre_tokenizer_process(self, text: str) -> List[str]:
        return [text]

    @property
    def vocab_size(self) -> int:
        return self.n_words

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.encoder)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return [self.decoder[t] for t in self.encode(text)]

    def _convert_token_to_id(self, token: str) -> Optional[int]:
        return self.encoder.get(token, self.unk_id)

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self.decoder.get(index)

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        return out_string

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        return bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", "replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            raise ValueError(f"vocabulary path ({save_directory}) should be a directory")
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
