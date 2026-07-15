"""Shadow-only HIMA policy subagent and local Transformers text generation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol

from rtscortex.policy.hima.models import HIMA_ADAPTER_VERSION
from rtscortex.policy.hima.observation import HIMAObservationAdapter
from rtscortex.policy.hima.parser import HIMAProposalParser
from rtscortex.policy.models import (
    MacroPolicyProposal,
    PolicyGenerationMetadata,
    PolicyObservationFixture,
    PolicyProviderKind,
    PolicySubagentSpec,
)

HIMA_PINNED_REVISIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "SNUMPR/Protoss-a": "95348eea419b2e2d9717d747ca30e05a0cba787d",
        "SNUMPR/Protoss-b": "6b0faaf7c3f7a9544d9fa595a8077cc72a8747a6",
        "SNUMPR/Protoss-c": "09ae752994bcd458d3b91f5b97dcdacd626edccb",
    }
)


class HIMATextGenerator(Protocol):
    """Generate one HIMA response from one user-only observation message."""

    async def generate(self, *, user_message: str) -> str: ...


class HIMAPolicySubagent:
    """Adapt, generate and parse a HIMA macro proposal without dispatching it."""

    def __init__(
        self,
        spec: PolicySubagentSpec,
        generator: HIMATextGenerator,
        adapter: HIMAObservationAdapter,
        parser: HIMAProposalParser,
    ) -> None:
        if spec.provider_kind is not PolicyProviderKind.HUGGING_FACE_TRANSFORMERS:
            raise ValueError("HIMA subagents require the Hugging Face Transformers provider")
        if spec.model_id not in HIMA_PINNED_REVISIONS:
            raise ValueError(f"unrecognized pinned HIMA model: {spec.model_id}")
        if spec.race != "Protoss":
            raise ValueError("HIMA Protoss subagents require race='Protoss'")
        self.spec = spec
        self.generator = generator
        self.adapter = adapter
        self.parser = parser

    @property
    def model_revision(self) -> str:
        """Return immutable model provenance for comparison artifacts."""

        return HIMA_PINNED_REVISIONS[self.spec.model_id]

    async def propose(self, fixture: PolicyObservationFixture) -> MacroPolicyProposal:
        """Return a parsed advisory macro proposal for one immutable fixture."""

        _, user_message = self.adapter.prepare(fixture)
        raw_output = await self.generator.generate(user_message=user_message)
        metadata = getattr(self.generator, "last_generation_metadata", None)
        if metadata is not None and not isinstance(metadata, PolicyGenerationMetadata):
            raise TypeError("HIMA generation metadata has an invalid type")
        if metadata is not None:
            expected_revision = HIMA_PINNED_REVISIONS[self.spec.model_id]
            if (
                metadata.provider_kind
                is not PolicyProviderKind.HUGGING_FACE_TRANSFORMERS
                or metadata.model_id != self.spec.model_id
                or metadata.model_revision != expected_revision
                or not metadata.checkpoint_verified
                or not metadata.license_acknowledged
            ):
                raise ValueError("HIMA generation metadata does not match the pinned spec")
        proposal = self.parser.parse(
            raw_output,
            truncated=bool(metadata is not None and metadata.truncated),
        )
        return proposal.model_copy(
            update={
                "adapter_version": HIMA_ADAPTER_VERSION,
                "generation_metadata": metadata,
            }
        )


class TransformersHIMAGenerator:
    """Lazy, local-only Transformers generator for one pinned HIMA checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str,
        allow_unlicensed_weights: bool,
        device: str = "cuda",
        max_new_tokens: int = 2048,
    ) -> None:
        if not allow_unlicensed_weights:
            raise PermissionError(
                "HIMA weights have no declared license; set allow_unlicensed_weights=true "
                "only after accepting that risk"
            )
        if model_id not in HIMA_PINNED_REVISIONS:
            raise ValueError(f"unrecognized pinned HIMA model: {model_id}")

        expanded = Path(model_path).expanduser()
        if not expanded.is_absolute():
            raise ValueError("HIMA model_path must be an explicit absolute local path")
        if not expanded.is_dir():
            raise FileNotFoundError(f"local HIMA model directory does not exist: {expanded}")

        expected_revision = HIMA_PINNED_REVISIONS[model_id]
        self.model_path = _verified_snapshot_path(expanded, model_id, expected_revision)
        self.model_id = model_id
        self.model_revision = expected_revision
        self.device = device
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.max_new_tokens = max_new_tokens
        self.last_generation_metadata: PolicyGenerationMetadata | None = None
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    async def generate(self, *, user_message: str) -> str:
        """Generate deterministically off the event loop using a single user message."""

        if not user_message:
            raise ValueError("HIMA user_message must not be empty")
        return await asyncio.to_thread(self._generate_sync, user_message)

    def _generate_sync(self, user_message: str) -> str:
        torch, tokenizer, model = self._load()
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise TypeError("HIMA tokenizer chat template must render text")

        encoded = tokenizer(rendered, return_tensors="pt").to(self.device)
        input_length = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        continuation = generated[0][input_length:]
        completion_token_count = _token_count(continuation)
        eos_reached = _last_token_id(continuation) in _eos_token_ids(model, tokenizer)
        truncated = completion_token_count >= self.max_new_tokens and not eos_reached
        decoded = tokenizer.decode(continuation, skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise TypeError("HIMA tokenizer decode must return text")
        self.last_generation_metadata = PolicyGenerationMetadata(
            provider_kind=PolicyProviderKind.HUGGING_FACE_TRANSFORMERS,
            model_id=self.model_id,
            model_revision=self.model_revision,
            checkpoint_path=str(self.model_path),
            checkpoint_verified=True,
            license_acknowledged=True,
            deterministic=True,
            max_new_tokens=self.max_new_tokens,
            prompt_token_count=input_length,
            completion_token_count=completion_token_count,
            eos_reached=eos_reached,
            truncated=truncated,
        )
        return decoded.strip()

    def _load(self) -> tuple[Any, Any, Any]:
        if self._torch is not None and self._tokenizer is not None and self._model is not None:
            return self._torch, self._tokenizer, self._model

        try:
            torch = import_module("torch")
            transformers = import_module("transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "local HIMA inference requires the optional torch and transformers packages"
            ) from exc

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            torch_dtype=torch.float16,
        )
        model.eval()
        model.to(self.device)

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        return torch, tokenizer, model


def _verified_snapshot_path(
    path: Path,
    model_id: str,
    expected_revision: str,
) -> Path:
    resolved = path.resolve()
    expected_repo_token = f"models--{model_id.replace('/', '--')}"
    if (
        resolved.parent.name == "snapshots"
        and resolved.parent.parent.name == expected_repo_token
        and resolved.name == expected_revision
    ):
        return resolved

    raise ValueError(
        "HIMA model_path must resolve to the pinned Hugging Face snapshot "
        f"{expected_repo_token}/snapshots/{expected_revision}"
    )


def _token_count(tokens: Any) -> int:
    shape = getattr(tokens, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(tokens)


def _last_token_id(tokens: Any) -> int | None:
    if _token_count(tokens) == 0:
        return None
    value = tokens[-1]
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _eos_token_ids(model: Any, tokenizer: Any) -> frozenset[int]:
    generation_config = getattr(model, "generation_config", None)
    raw_ids = getattr(generation_config, "eos_token_id", None)
    if raw_ids is None:
        raw_ids = getattr(tokenizer, "eos_token_id", None)
    if raw_ids is None:
        return frozenset()
    if isinstance(raw_ids, int):
        return frozenset({raw_ids})
    return frozenset(int(value) for value in raw_ids)
