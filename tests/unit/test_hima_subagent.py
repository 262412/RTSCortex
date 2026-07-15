from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import rtscortex.policy.hima.subagent as hima_subagent_module
from rtscortex.policy.hima.observation import HIMAObservationAdapter
from rtscortex.policy.hima.parser import HIMAProposalParser
from rtscortex.policy.hima.subagent import (
    HIMA_PINNED_REVISIONS,
    HIMAPolicySubagent,
    TransformersHIMAGenerator,
)
from rtscortex.policy.models import (
    PolicyGenerationMetadata,
    PolicyObservationFixture,
    PolicyProviderKind,
)
from rtscortex.policy.subagents import HIMA_PROTOSS_SPECS, QWEN3_8B_SPEC
from tests.helpers import make_observation


class RecordingGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.user_messages: list[str] = []

    async def generate(self, *, user_message: str) -> str:
        self.user_messages.append(user_message)
        return self.response


def _snapshot_path(tmp_path: Path, model_id: str) -> Path:
    revision = HIMA_PINNED_REVISIONS[model_id]
    path = (
        tmp_path
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    path.mkdir(parents=True)
    return path


def test_hima_subagent_adapts_one_user_message_and_parses_macro_proposal() -> None:
    generator = RecordingGenerator(
        "Final Actions Summary: <TRAIN PROBE> x 4 <BUILD PYLON>"
    )
    subagent = HIMAPolicySubagent(
        HIMA_PROTOSS_SPECS[0],
        generator,
        HIMAObservationAdapter(),
        HIMAProposalParser(),
    )
    fixture = PolicyObservationFixture(
        fixture_id="opening",
        observation=make_observation(game_loop=224),
        previous_actions=["Probe", "Pylon"],
    )

    proposal = asyncio.run(subagent.propose(fixture))

    assert len(generator.user_messages) == 1
    prompt = json.loads(generator.user_messages[0])
    assert list(prompt) == [
        "supply_used",
        "supply_capacity",
        "unit",
        "research",
        "previous_action",
    ]
    assert prompt["previous_action"] == ["Probe", "Pylon"]
    assert "game_time" not in prompt
    assert "production" not in prompt
    assert "available_actions" not in prompt
    assert [step.canonical_action for step in proposal.steps] == [
        "TRAIN PROBE",
        "BUILD PYLON",
    ]
    assert proposal.steps[0].repeat == 4
    assert proposal.raw_output.startswith("Final Actions Summary")
    assert proposal.adapter_version == "hima-github-json-v2"
    assert proposal.generation_metadata is None
    assert subagent.model_revision == HIMA_PINNED_REVISIONS["SNUMPR/Protoss-a"]


def test_hima_subagent_rejects_non_hima_or_unpinned_specs() -> None:
    generator = RecordingGenerator("Actions: []")
    with pytest.raises(ValueError, match="Hugging Face Transformers"):
        HIMAPolicySubagent(
            QWEN3_8B_SPEC,
            generator,
            HIMAObservationAdapter(),
            HIMAProposalParser(),
        )

    unknown = HIMA_PROTOSS_SPECS[0].model_copy(update={"model_id": "local/unknown"})
    with pytest.raises(ValueError, match="unrecognized pinned HIMA model"):
        HIMAPolicySubagent(
            unknown,
            generator,
            HIMAObservationAdapter(),
            HIMAProposalParser(),
        )


def test_pinned_hima_revisions_cover_all_three_official_candidates() -> None:
    assert dict(HIMA_PINNED_REVISIONS) == {
        "SNUMPR/Protoss-a": "95348eea419b2e2d9717d747ca30e05a0cba787d",
        "SNUMPR/Protoss-b": "6b0faaf7c3f7a9544d9fa595a8077cc72a8747a6",
        "SNUMPR/Protoss-c": "09ae752994bcd458d3b91f5b97dcdacd626edccb",
    }


def test_transformers_generator_enforces_license_and_explicit_local_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(PermissionError, match="no declared license"):
        TransformersHIMAGenerator(
            missing,
            model_id="SNUMPR/Protoss-a",
            allow_unlicensed_weights=False,
        )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        TransformersHIMAGenerator(
            missing,
            model_id="SNUMPR/Protoss-a",
            allow_unlicensed_weights=True,
        )
    with pytest.raises(ValueError, match="absolute local path"):
        TransformersHIMAGenerator(
            Path("relative-checkpoint"),
            model_id="SNUMPR/Protoss-a",
            allow_unlicensed_weights=True,
        )


def test_transformers_generator_requires_exact_model_snapshot_identity(
    tmp_path: Path,
) -> None:
    revision = HIMA_PINNED_REVISIONS["SNUMPR/Protoss-a"]
    fake_repo = tmp_path / "models--other--model" / "snapshots" / revision
    fake_repo.mkdir(parents=True)

    with pytest.raises(ValueError, match="models--SNUMPR--Protoss-a"):
        TransformersHIMAGenerator(
            fake_repo,
            model_id="SNUMPR/Protoss-a",
            allow_unlicensed_weights=True,
        )

    exact = _snapshot_path(tmp_path, "SNUMPR/Protoss-a")
    generator = TransformersHIMAGenerator(
        exact,
        model_id="SNUMPR/Protoss-a",
        allow_unlicensed_weights=True,
    )
    assert generator.model_path == exact.resolve()


class FakeEncoded(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__(input_ids=SimpleNamespace(shape=(1, 3)))
        self.device: str | None = None

    def to(self, device: str) -> FakeEncoded:
        self.device = device
        return self


class FakeGeneratedRow:
    def __getitem__(self, item: slice) -> list[int]:
        assert item.start == 3
        return [8, 9]


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None
        self.rendered_prompt: str | None = None
        self.encoded = FakeEncoded()
        self.eos_token_id = 9

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        self.messages = messages
        return "rendered user prompt"

    def __call__(self, rendered: str, *, return_tensors: str) -> FakeEncoded:
        assert return_tensors == "pt"
        self.rendered_prompt = rendered
        return self.encoded

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert tokens == [8, 9]
        assert skip_special_tokens
        return "  Actions: ['Probe', 'Pylon']  "


class FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.device: str | None = None
        self.generate_kwargs: dict[str, Any] = {}

    def eval(self) -> None:
        self.eval_called = True

    def to(self, device: str) -> None:
        self.device = device

    def generate(self, **kwargs: Any) -> list[FakeGeneratedRow]:
        self.generate_kwargs = kwargs
        return [FakeGeneratedRow()]


class FakeFactory:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, path: str, **kwargs: object) -> object:
        self.calls.append((path, kwargs))
        return self.value


def test_transformers_generator_is_lazy_local_only_user_only_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    tokenizer_factory = FakeFactory(tokenizer)
    model_factory = FakeFactory(model)
    fake_float16 = object()
    fake_torch = SimpleNamespace(
        float16=fake_float16,
        inference_mode=lambda: nullcontext(),
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=tokenizer_factory,
        AutoModelForCausalLM=model_factory,
    )
    imports: list[str] = []

    def fake_import(name: str) -> object:
        imports.append(name)
        return fake_torch if name == "torch" else fake_transformers

    monkeypatch.setattr(hima_subagent_module, "import_module", fake_import)
    snapshot = _snapshot_path(tmp_path, "SNUMPR/Protoss-b")
    generator = TransformersHIMAGenerator(
        snapshot,
        model_id="SNUMPR/Protoss-b",
        allow_unlicensed_weights=True,
        device="cuda:1",
    )

    assert imports == []
    result = asyncio.run(generator.generate(user_message='{"supply_used":12}'))

    assert result == "Actions: ['Probe', 'Pylon']"
    assert imports == ["torch", "transformers"]
    assert tokenizer.messages == [
        {"role": "user", "content": '{"supply_used":12}'}
    ]
    assert all(message["role"] != "system" for message in tokenizer.messages)
    assert tokenizer_factory.calls == [
        (str(snapshot), {"local_files_only": True})
    ]
    assert model_factory.calls == [
        (
            str(snapshot),
            {"local_files_only": True, "torch_dtype": fake_float16},
        )
    ]
    assert model.eval_called
    assert model.device == "cuda:1"
    assert tokenizer.encoded.device == "cuda:1"
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 2048
    assert generator.model_revision == HIMA_PINNED_REVISIONS["SNUMPR/Protoss-b"]
    assert generator.last_generation_metadata == PolicyGenerationMetadata(
        provider_kind=PolicyProviderKind.HUGGING_FACE_TRANSFORMERS,
        model_id="SNUMPR/Protoss-b",
        model_revision=HIMA_PINNED_REVISIONS["SNUMPR/Protoss-b"],
        checkpoint_path=str(snapshot),
        checkpoint_verified=True,
        license_acknowledged=True,
        deterministic=True,
        max_new_tokens=2048,
        prompt_token_count=3,
        completion_token_count=2,
        eos_reached=True,
        truncated=False,
    )

    asyncio.run(generator.generate(user_message='{"supply_used":13}'))
    assert imports == ["torch", "transformers"]


def test_transformers_generator_reports_missing_optional_packages_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(hima_subagent_module, "import_module", missing_import)
    snapshot = _snapshot_path(tmp_path, "SNUMPR/Protoss-c")
    generator = TransformersHIMAGenerator(
        snapshot,
        model_id="SNUMPR/Protoss-c",
        allow_unlicensed_weights=True,
    )

    with pytest.raises(RuntimeError, match="optional torch and transformers"):
        asyncio.run(generator.generate(user_message="{}"))


def test_truncated_generation_is_recorded_and_reaches_proposal_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = 999
    model = FakeModel()
    fake_torch = SimpleNamespace(
        float16=object(),
        inference_mode=lambda: nullcontext(),
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeFactory(tokenizer),
        AutoModelForCausalLM=FakeFactory(model),
    )

    def fake_import(name: str) -> object:
        return fake_torch if name == "torch" else fake_transformers

    monkeypatch.setattr(hima_subagent_module, "import_module", fake_import)
    snapshot = _snapshot_path(tmp_path, "SNUMPR/Protoss-b")
    generator = TransformersHIMAGenerator(
        snapshot,
        model_id="SNUMPR/Protoss-b",
        allow_unlicensed_weights=True,
        max_new_tokens=2,
    )
    subagent = HIMAPolicySubagent(
        HIMA_PROTOSS_SPECS[1],
        generator,
        HIMAObservationAdapter(),
        HIMAProposalParser(),
    )

    proposal = asyncio.run(
        subagent.propose(
            PolicyObservationFixture(
                fixture_id="truncated",
                observation=make_observation(),
            )
        )
    )

    assert proposal.generation_metadata is not None
    assert proposal.generation_metadata.truncated is True
    assert proposal.generation_metadata.eos_reached is False
    assert [item.code for item in proposal.diagnostics] == ["output_truncated"]


class ProvenanceMismatchGenerator(RecordingGenerator):
    def __init__(self) -> None:
        super().__init__("Actions: ['Probe']")
        self.last_generation_metadata = PolicyGenerationMetadata(
            provider_kind=PolicyProviderKind.HUGGING_FACE_TRANSFORMERS,
            model_id="SNUMPR/Protoss-b",
            model_revision=HIMA_PINNED_REVISIONS["SNUMPR/Protoss-b"],
            checkpoint_path="/models--SNUMPR--Protoss-b/snapshots/revision",
            checkpoint_verified=True,
            license_acknowledged=True,
            deterministic=True,
            max_new_tokens=16,
            prompt_token_count=4,
            completion_token_count=2,
            eos_reached=True,
            truncated=False,
        )


def test_subagent_rejects_generation_provenance_for_another_checkpoint() -> None:
    subagent = HIMAPolicySubagent(
        HIMA_PROTOSS_SPECS[0],
        ProvenanceMismatchGenerator(),
        HIMAObservationAdapter(),
        HIMAProposalParser(),
    )

    with pytest.raises(ValueError, match="does not match the pinned spec"):
        asyncio.run(
            subagent.propose(
                PolicyObservationFixture(
                    fixture_id="mismatch",
                    observation=make_observation(),
                )
            )
        )
