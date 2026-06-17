"""Self-contained, engine-agnostic multi-LoRA RL recipe.

A single-file reference for RL-style training of MANY LoRA adapters over one
shared base model, with rollouts served by a *pluggable* inference engine. No
sibling imports, no external session JSON — everything needed is inline.

Flow (drives the XORL training server's public REST API + a rollout engine):

    1. create_model               one LoRA adapter (rank/alpha/optimizer) per session
    2. forward_backward + optim   train each adapter for a few steps
    3. save_weights_for_sampler   export each adapter to disk (HF/PEFT; per-expert
                                  stacked A/B for routed-expert MoE)
    4. engine.register_adapter    make the exported adapter servable, by id
    5. engine.generate            per-adapter rollout (text + per-token logprobs)

    Repeat 2-5 for an RL loop. Steps 3-4 are export+reload today; an in-place
    *per-adapter weight refit* (push only the changed adapter's A/B, no disk) is
    the performance gap for a true RL inner loop — see RolloutEngine.refit_adapter.

Two engines implement the same `RolloutEngine` interface:

    - SGLangEngine : the path validated end-to-end. Exports are pushed to SGLang
                     via the training server's create_sampling_session, and
                     rollouts hit SGLang's HTTP /generate (per-request `lora_path`,
                     `return_logprob`).
    - TRTLLMEngine : reference implementation against TensorRT-LLM's in-process
                     `LLM` API + `LoRARequest` + `SamplingParams`. NOT executed
                     here; see the constraints noted on the class.

Setup — start the prerequisite servers, then run this recipe:

    1. XORL training server. Launch with a LoRA server config from this repo
       (examples/server/configs/lora/qwen3_30b_a3b_lora.yaml: Qwen3-30B-A3B,
       routed-expert multi-LoRA, EP=8 -> 8 GPUs; add
       --server.expert_parallel_size 4 to fit fewer GPUs):

         python -m xorl.server.launcher --mode auto \
             --config examples/server/configs/lora/qwen3_30b_a3b_lora.yaml \
             --api-port 26000

    2. Rollout engine.
       (a) SGLang -- launch a LoRA-enabled server on the same base model:

             python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B \
                 --tp 2 --dtype bfloat16 --port 30000 --trust-remote-code \
                 --enable-lora --max-lora-rank 16 --max-loras-per-batch 4 \
                 --max-loaded-loras 8 --lora-moe-format per_expert \
                 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

       (b) TensorRT-LLM -- no separate server; this recipe builds the in-process
           LLM itself (just pass --engine trtllm --base-model <path>).

    3. Run this recipe. --base-model must match the training server's model_path
       (and, for SGLang, --model-path):

         # SGLang
         python multilora_rl_recipe.py --engine sglang \
             --train-url http://localhost:26000 \
             --inference-url http://localhost:30000 \
             --base-model Qwen/Qwen3-30B-A3B

         # TensorRT-LLM (in-process; no --inference-url)
         python multilora_rl_recipe.py --engine trtllm \
             --train-url http://localhost:26000 \
             --base-model /path/to/Qwen3-30B-A3B
"""

from __future__ import annotations

import abc
import argparse
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are a strict lookup assistant. When asked for an adapter-specific unlock code, "
    "respond with exactly the stored code."
)

# Inline session specs (replaces the external JSON). Each is one LoRA adapter with
# its own rank / alpha / optimizer — heterogeneous on purpose to exercise the
# multi-adapter path. Trim with --num-adapters.
DEFAULT_SESSIONS: list[dict[str, Any]] = [
    {
        "name": "signsgd-r4",
        "lora_rank": 4,
        "lora_alpha": 8,
        "optimizer_config": {"type": "signsgd", "learning_rate": 2e-4, "weight_decay": 0.0},
    },
    {
        "name": "sgd-r8",
        "lora_rank": 8,
        "lora_alpha": 16,
        "optimizer_config": {
            "type": "sgd",
            "learning_rate": 5e-4,
            "weight_decay": 0.0,
            "optimizer_kwargs": {"momentum": 0.9, "nesterov": True},
        },
    },
    {
        "name": "adamw-r16",
        "lora_rank": 16,
        "lora_alpha": 32,
        "optimizer_config": {
            "type": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
    },
]


# ---------------------------------------------------------------------------
# Training server client (XORL public async REST API)
# ---------------------------------------------------------------------------
class TrainingServerClient:
    """Minimal requests-based client for the XORL training server.

    The server is async: POST returns a ``request_id``; poll ``/retrieve_future``
    until the result is ready.
    """

    def __init__(
        self,
        base_url: str,
        *,
        future_timeout: float = 1800.0,
        future_poll_interval: float = 0.5,
        request_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.future_timeout = future_timeout
        self.future_poll_interval = future_poll_interval
        self.request_retries = max(0, int(request_retries))
        self.session = requests.Session()
        self.session.headers.update({"Connection": "close"})

    def close(self) -> None:
        self.session.close()

    def _post(self, path: str, payload: dict[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
        last: Optional[Exception] = None
        for attempt in range(self.request_retries + 1):
            try:
                r = self.session.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                last = exc
                if attempt >= self.request_retries:
                    raise
                time.sleep(min(0.25 * (attempt + 1), 1.0))
        raise last or RuntimeError(f"POST {path} failed")

    def _get(self, path: str, *, timeout: float = 30.0) -> dict[str, Any]:
        r = self.session.get(f"{self.base_url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()

    def wait_for_service(self, *, timeout: float = 1800.0, poll_interval: float = 3.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                payload = self.session.get(f"{self.base_url}/health", timeout=5).json()
                if payload.get("engine_running"):
                    return
            except Exception:
                pass
            time.sleep(poll_interval)
        raise TimeoutError(f"training server {self.base_url} not healthy within {timeout:.0f}s")

    def _await_future(self, request_id: str, *, context: str) -> dict[str, Any]:
        deadline = time.time() + self.future_timeout
        while time.time() < deadline:
            result = self._post("/api/v1/retrieve_future", {"request_id": request_id})
            if result.get("type") == "try_again":
                time.sleep(self.future_poll_interval)
                continue
            if result.get("type") == "request_failed" or ("error" in result and "category" in result):
                raise RuntimeError(f"{context} failed: {result.get('error', result)}")
            return result
        raise TimeoutError(f"{context} future {request_id} timed out")

    def _submit(self, path: str, payload: dict[str, Any], *, context: str) -> dict[str, Any]:
        future = self._post(path, payload)
        request_id = future.get("request_id")
        if not request_id:
            raise RuntimeError(f"{context} returned no request_id: {future}")
        return self._await_future(request_id, context=context)

    # -- adapter training --------------------------------------------------
    def create_model(
        self, *, model_id: str, base_model: str, lora_rank: int, lora_alpha: int, optimizer_config: dict[str, Any]
    ) -> dict[str, Any]:
        return self._submit(
            "/api/v1/create_model",
            {
                "model_id": model_id,
                "base_model": base_model,
                "lora_config": {
                    "rank": lora_rank,
                    "lora_rank": lora_rank,
                    "alpha": lora_alpha,
                    "lora_alpha": lora_alpha,
                },
                "optimizer_config": optimizer_config,
            },
            context=f"create_model({model_id})",
        )

    def forward(self, *, model_id: str, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return self._submit(
            "/api/v1/forward",
            {
                "model_id": model_id,
                "forward_input": {"data": batch, "loss_fn": "causallm_loss"},
            },
            context=f"forward({model_id})",
        )

    def forward_backward(self, *, model_id: str, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return self._submit(
            "/api/v1/forward_backward",
            {
                "model_id": model_id,
                "forward_backward_input": {"data": batch, "loss_fn": "causallm_loss"},
            },
            context=f"forward_backward({model_id})",
        )

    def optim_step(self, *, model_id: str, gradient_clip: float) -> dict[str, Any]:
        return self._submit(
            "/api/v1/optim_step",
            {
                "model_id": model_id,
                "gradient_clip": gradient_clip,
            },
            context=f"optim_step({model_id})",
        )

    def save_weights_for_sampler(self, *, model_id: str, name: str) -> dict[str, Any]:
        return self._submit(
            "/api/v1/save_weights_for_sampler",
            {
                "model_id": model_id,
                "name": name,
            },
            context=f"save_weights_for_sampler({model_id})",
        )

    def unload_model(self, *, model_id: str) -> dict[str, Any]:
        return self._submit("/api/v1/unload_model", {"model_id": model_id}, context=f"unload_model({model_id})")

    # -- SGLang serving bridge (used only by SGLangEngine) -----------------
    def add_inference_endpoint(self, *, host: str, port: int) -> dict[str, Any]:
        return self._post(
            "/add_inference_endpoint",
            {
                "host": host,
                "port": port,
                "worker_port": port,
                "world_size": 1,
                "sync_weights": False,
            },
        )

    def create_sampling_session(self, *, model_path: str) -> dict[str, Any]:
        return self._post("/api/v1/create_sampling_session", {"model_path": model_path})


# ---------------------------------------------------------------------------
# Synthetic per-adapter SFT data (deterministic: each adapter memorizes its codes)
# ---------------------------------------------------------------------------
def _chat_ids(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    kwargs = {"tokenize": True, "add_generation_prompt": add_generation_prompt, "return_dict": False}
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    return list(ids)


def _code(adapter_name: str, i: int) -> str:
    d = hashlib.sha256(f"{adapter_name}:{i}".encode()).hexdigest().upper()
    return f"{d[:4]}-{d[4:8]}-{d[8:12]}"


def build_adapter_dataset(tokenizer: Any, *, adapter_name: str, num_examples: int) -> list[dict[str, Any]]:
    """Deterministic SFT examples for one adapter; labels mask the prompt (-100)."""
    dataset: list[dict[str, Any]] = []
    for i in range(num_examples):
        item = f"{adapter_name.replace('-', '_')}_artifact_{i:03d}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What is the unlock code for {item}?"},
            {"role": "assistant", "content": _code(adapter_name, i)},
        ]
        full_ids = _chat_ids(tokenizer, messages, add_generation_prompt=False)
        prompt_ids = _chat_ids(tokenizer, messages[:2], add_generation_prompt=True)
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        dataset.append({"model_input": {"input_ids": full_ids}, "loss_fn_inputs": {"labels": labels}})
    return dataset


def _select_batch(dataset: list[dict[str, Any]], *, step: int, batch_size: int) -> list[dict[str, Any]]:
    return [dataset[(step * batch_size + o) % len(dataset)] for o in range(batch_size)]


def _loss_mean(fb_result: dict[str, Any]) -> float:
    metrics = fb_result.get("metrics", {})
    if "loss:mean" in metrics:
        return float(metrics["loss:mean"])
    outputs = fb_result.get("loss_fn_outputs", [])
    if outputs and isinstance(outputs[0].get("loss"), dict):
        data = outputs[0]["loss"].get("data")
        if isinstance(data, list) and data:
            return float(data[0])
    raise KeyError(f"no loss:mean in forward_backward result: {fb_result}")


@dataclass
class Session:
    name: str
    model_id: str
    lora_rank: int
    lora_alpha: int
    optimizer_config: dict[str, Any]
    served_name: Optional[str] = None  # set after the engine registers the adapter


# ---------------------------------------------------------------------------
# Rollout engine interface + two implementations
# ---------------------------------------------------------------------------
@dataclass
class Rollout:
    text: str
    token_logprobs: list[float] = field(default_factory=list)


class RolloutEngine(abc.ABC):
    """What the RL loop needs from the rollout engine.

    The trainer (XORL) is engine-independent; only adapter serving + sampling
    differ between SGLang and TensorRT-LLM, and live behind this interface.
    """

    @abc.abstractmethod
    def register_adapter(self, *, name: str, exported_path: str) -> str:
        """Make an exported adapter (from save_weights_for_sampler) servable.
        Returns the id used to select it per request."""

    @abc.abstractmethod
    def generate(
        self, *, prompt: str, adapter: str, max_tokens: int, temperature: float, return_logprobs: bool
    ) -> Rollout:
        """Sample a rollout from a specific resident adapter."""

    def refit_adapter(self, *, name: str, exported_path: str) -> str:
        """In-place per-adapter weight update for an RL inner loop.

        GAP: neither engine supports an in-place per-adapter weight refit today,
        so the default falls back to register_adapter (export+reload). Replace
        this with a real per-adapter push (no disk) when the engine supports it.
        """
        return self.register_adapter(name=name, exported_path=exported_path)

    def close(self) -> None:  # optional
        pass


class SGLangEngine(RolloutEngine):
    """Validated path. Adapters are pushed to SGLang by the training server
    (create_sampling_session -> SGLang /load_lora_adapter); rollouts hit SGLang's
    HTTP /generate with a per-request ``lora_path`` and ``return_logprob``."""

    def __init__(self, *, train_client: TrainingServerClient, inference_url: str):
        self.client = train_client
        self.inference_url = inference_url.rstrip("/")
        parsed = urlparse(inference_url)
        if not parsed.hostname or not parsed.port:
            raise ValueError(f"--inference-url needs host:port, got {inference_url}")
        # Register the SGLang endpoint with the training server so it can push adapters.
        self.client.add_inference_endpoint(host=parsed.hostname, port=parsed.port)
        # Wait for SGLang to be serving.
        deadline = time.time() + 1800
        while time.time() < deadline:
            try:
                if requests.get(f"{self.inference_url}/v1/models", timeout=10).json().get("data"):
                    break
            except Exception:
                pass
            time.sleep(5)

    def register_adapter(self, *, name: str, exported_path: str) -> str:
        result = self.client.create_sampling_session(model_path=exported_path)
        return str(result["lora_name"])

    def generate(
        self, *, prompt: str, adapter: str, max_tokens: int, temperature: float, return_logprobs: bool
    ) -> Rollout:
        payload = {
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": temperature},
            "lora_path": adapter,  # per-request adapter selection
            "return_logprob": return_logprobs,
        }
        resp = requests.post(f"{self.inference_url}/generate", json=payload, timeout=300).json()
        if isinstance(resp, list):
            resp = resp[0]
        logprobs: list[float] = []
        if return_logprobs:
            meta = resp.get("meta_info", {}) or {}
            logprobs = [float(lp[0]) for lp in (meta.get("output_token_logprobs") or [])]
        return Rollout(text=resp.get("text", ""), token_logprobs=logprobs)


class TRTLLMEngine(RolloutEngine):
    """Reference implementation against TensorRT-LLM's in-process LLM API.

    NOT executed in this repo's validation — provided so the same recipe runs
    against TRT-LLM. Maps 1:1 onto documented APIs:
      - adapter selection  : LoRARequest(lora_name, lora_int_id, lora_path)
      - sampling + logprobs: SamplingParams(max_tokens, temperature, logprobs=0)
      - generation         : LLM.generate(prompt, sampling_params, lora_request=...)

    Routed-expert (MoE) LoRA constraints (TRT-LLM docs/features/lora.md):
      - moe_backend="CUTLASS" only; base weights bf16/fp16 (no FP8/FP4/INT yet);
        adapters are per-expert PEFT (stacked [num_experts, ...]) — which is what
        save_weights_for_sampler exports. DoRA / min-latency / alltoall unsupported.
    """

    def __init__(self, *, base_model: str, max_lora_rank: int = 16, max_loras: int = 8):
        from tensorrt_llm import LLM  # noqa: PLC0415  (lazy: tensorrt_llm only needed for this engine)
        from tensorrt_llm.lora_manager import LoraConfig  # noqa: PLC0415

        self.LLM = LLM
        self.llm = LLM(
            model=base_model,
            moe_backend="CUTLASS",  # required for routed-expert MoE LoRA
            lora_config=LoraConfig(
                lora_target_modules=[
                    "attn_q",
                    "attn_k",
                    "attn_v",
                    "attn_dense",
                    "moe_h_to_4h",
                    "moe_4h_to_h",
                    "moe_gate",  # routed-expert LoRA
                ],
                max_lora_rank=max_lora_rank,
                max_loras=max_loras,
                max_cpu_loras=max_loras,
            ),
        )
        self._next_id = 1
        self._adapters: dict[str, Any] = {}

    def register_adapter(self, *, name: str, exported_path: str) -> str:
        from tensorrt_llm.executor import LoRARequest  # noqa: PLC0415

        # exported_path must be a local dir readable by this process; resolve the
        # training server's xorl:// URI to its absolute path if necessary.
        self._adapters[name] = LoRARequest(name, self._next_id, exported_path)
        self._next_id += 1
        return name

    def generate(
        self, *, prompt: str, adapter: str, max_tokens: int, temperature: float, return_logprobs: bool
    ) -> Rollout:
        from tensorrt_llm import SamplingParams  # noqa: PLC0415

        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=0 if return_logprobs else None,  # 0 => sampled-token logprob
        )
        out = self.llm.generate(prompt, sp, lora_request=self._adapters[adapter]).outputs[0]
        logprobs: list[float] = []
        if return_logprobs and out.logprobs:
            for entry in out.logprobs:
                # SimpleTokenLogprobs (list[float]) or list[dict[token_id, Logprob]]
                if isinstance(entry, dict):
                    logprobs.append(float(next(iter(entry.values())).logprob))
                else:
                    logprobs.append(float(entry))
        return Rollout(text=out.text, token_logprobs=logprobs)

    def close(self) -> None:
        try:
            self.llm.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["sglang", "trtllm"], required=True)
    p.add_argument("--train-url", required=True, help="XORL training server base URL")
    p.add_argument("--inference-url", help="SGLang base URL (required for --engine sglang)")
    p.add_argument("--base-model", required=True, help="HF id or path of the shared base model")
    p.add_argument("--num-adapters", type=int, default=3)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--examples-per-adapter", type=int, default=8)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def build_engine(args: argparse.Namespace, client: TrainingServerClient) -> RolloutEngine:
    if args.engine == "sglang":
        if not args.inference_url:
            raise SystemExit("--inference-url is required for --engine sglang")
        return SGLangEngine(train_client=client, inference_url=args.inference_url)
    return TRTLLMEngine(base_model=args.base_model)


def main() -> int:
    args = parse_args()
    run_id = args.run_id or time.strftime("multilora-%Y%m%dT%H%M%SZ", time.gmtime())

    sessions = [
        Session(
            name=s["name"],
            model_id=f"{run_id}-{s['name']}",
            lora_rank=s["lora_rank"],
            lora_alpha=s["lora_alpha"],
            optimizer_config=s["optimizer_config"],
        )
        for s in DEFAULT_SESSIONS[: args.num_adapters]
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    datasets = {
        s.name: build_adapter_dataset(tokenizer, adapter_name=s.name, num_examples=args.examples_per_adapter)
        for s in sessions
    }

    client = TrainingServerClient(args.train_url)
    try:
        client.wait_for_service()
        engine = build_engine(args, client)

        # 1. create one LoRA adapter per session (+ one forward to force init)
        for s in sessions:
            print(f"[create] {s.name} (rank={s.lora_rank}, optimizer={s.optimizer_config['type']})")
            client.create_model(
                model_id=s.model_id,
                base_model=args.base_model,
                lora_rank=s.lora_rank,
                lora_alpha=s.lora_alpha,
                optimizer_config=s.optimizer_config,
            )
            client.forward(model_id=s.model_id, batch=[datasets[s.name][0]])

        # 2. train each adapter a few steps (interleaved)
        for step in range(args.steps):
            for s in sessions:
                batch = _select_batch(datasets[s.name], step=step, batch_size=args.batch_size)
                fb = client.forward_backward(model_id=s.model_id, batch=batch)
                opt = client.optim_step(model_id=s.model_id, gradient_clip=args.gradient_clip)
                print(
                    f"[train] {s.name} step={step:03d} loss={_loss_mean(fb):.6f} "
                    f"grad_norm={float(opt['metrics']['grad_norm']):.6f}"
                )

        # 3-5. export -> register on the engine -> per-adapter rollout
        for s in sessions:
            export = client.save_weights_for_sampler(model_id=s.model_id, name=f"{run_id}-{s.name}-sampler")
            s.served_name = engine.register_adapter(name=s.name, exported_path=str(export["path"]))
            rollout = engine.generate(
                prompt=f"What is the unlock code for {s.name.replace('-', '_')}_artifact_000?",
                adapter=s.served_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                return_logprobs=True,
            )
            print(
                f"[rollout] {s.name} served_as={s.served_name} "
                f"n_logprobs={len(rollout.token_logprobs)} text={rollout.text[:80]!r}"
            )

        print("\nmulti-LoRA RL recipe completed successfully")
        return 0
    finally:
        for s in sessions:
            try:
                client.unload_model(model_id=s.model_id)
            except Exception:
                pass
        try:
            engine.close()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
