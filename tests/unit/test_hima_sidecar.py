from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import cast

import pytest

from rtscortex.policy.hima import (
    HIMA_ADAPTER_VERSION,
    HIMA_PARSER_VERSION,
    HIMA_PINNED_REVISIONS,
    HIMA_VOCABULARY_VERSION,
    HIMALiveHealth,
    HIMALivePolicyClient,
)
from rtscortex.policy.hima.race_vocabulary import (
    HIMA_PARSER_VERSIONS,
    HIMA_VOCABULARY_VERSIONS,
)
from rtscortex.runtime.hima_sidecar import (
    HIMA_CANDIDATE_MODEL_IDS,
    HIMASidecarError,
    HIMASidecarProcess,
    HIMASidecarSpec,
    hima_sidecar_command,
    validate_local_hima_checkpoint,
    validate_sidecar_executable,
)


def _snapshot_path(root: Path, candidate: str) -> Path:
    model_id = HIMA_CANDIDATE_MODEL_IDS[candidate]
    revision = HIMA_PINNED_REVISIONS[model_id]
    path = root / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    path.mkdir(parents=True)
    return path


@pytest.mark.parametrize("candidate", tuple(HIMA_CANDIDATE_MODEL_IDS))
def test_validate_local_hima_checkpoint_locks_candidate_and_revision(
    tmp_path: Path,
    candidate: str,
) -> None:
    model_path = _snapshot_path(tmp_path, candidate)

    checkpoint = validate_local_hima_checkpoint(
        candidate=candidate,
        model_path=model_path,
        allow_unlicensed_weights=True,
    )

    assert checkpoint.model_id == HIMA_CANDIDATE_MODEL_IDS[candidate]
    assert checkpoint.revision == HIMA_PINNED_REVISIONS[checkpoint.model_id]
    assert checkpoint.model_path == model_path.resolve()


def test_validate_local_hima_checkpoint_requires_explicit_license_ack(
    tmp_path: Path,
) -> None:
    model_path = _snapshot_path(tmp_path, "protoss-a")

    with pytest.raises(HIMASidecarError, match="no declared license"):
        validate_local_hima_checkpoint(
            candidate="protoss-a",
            model_path=model_path,
            allow_unlicensed_weights=False,
        )


def test_validate_local_hima_checkpoint_rejects_swapped_snapshot(
    tmp_path: Path,
) -> None:
    model_path = _snapshot_path(tmp_path, "protoss-a")

    with pytest.raises(HIMASidecarError, match="provenance mismatch"):
        validate_local_hima_checkpoint(
            candidate="protoss-b",
            model_path=model_path,
            allow_unlicensed_weights=True,
        )


def test_hima_sidecar_command_is_explicit_and_local() -> None:
    command = hima_sidecar_command(
        Path("/opt/hima/bin/python"),
        socket_path=Path("/tmp/hima.sock"),
        model_id="SNUMPR/Protoss-a",
        model_path=Path("/models/hima-a"),
        device="cuda:1",
        max_new_tokens=256,
        allow_unlicensed_weights=True,
    )

    assert command[:3] == (
        "/opt/hima/bin/python",
        "-m",
        "rtscortex.policy.hima.live_worker",
    )
    assert "--socket" in command
    assert "--model-path" in command
    assert "--allow-unlicensed-weights" in command
    assert not any("huggingface.co" in argument for argument in command)


def test_validate_sidecar_executable_rejects_missing_python(tmp_path: Path) -> None:
    with pytest.raises(HIMASidecarError, match="missing or not executable"):
        validate_sidecar_executable((str(tmp_path / "missing-python"),))


class _FakeHealthClient(HIMALivePolicyClient):
    def __init__(self, health: HIMALiveHealth) -> None:
        self.health_payload = health
        self.closed = False

    async def health(self) -> HIMALiveHealth:
        return self.health_payload

    async def close(self) -> None:
        self.closed = True


class _HangingHealthClient(_FakeHealthClient):
    async def health(self) -> HIMALiveHealth:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _VanishedProcess:
    returncode: int | None = None

    def terminate(self) -> None:
        self.returncode = 0
        raise ProcessLookupError

    def kill(self) -> None:
        raise AssertionError("a vanished process must not be killed")

    async def wait(self) -> int:
        return 0


def test_hima_sidecar_process_waits_for_socket_and_cleans_up(tmp_path: Path) -> None:
    model_id = "SNUMPR/Protoss-a"
    revision = HIMA_PINNED_REVISIONS[model_id]
    socket_path = tmp_path / "hima.sock"
    script = tmp_path / "worker.py"
    script.write_text(
        "import os, pathlib, sys, time\n"
        "print(os.environ['HF_HUB_OFFLINE'], "
        "os.environ['TRANSFORMERS_OFFLINE'], flush=True)\n"
        "pathlib.Path(sys.argv[1]).touch()\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    client = _FakeHealthClient(
        HIMALiveHealth(
            model_id=model_id,
            model_revision=revision,
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version=HIMA_PARSER_VERSION,
            vocabulary_version=HIMA_VOCABULARY_VERSION,
        )
    )
    process = HIMASidecarProcess(
        HIMASidecarSpec(
            command=(sys.executable, str(script), str(socket_path)),
            socket_path=socket_path,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            ready_timeout_seconds=2.0,
            shutdown_timeout_seconds=2.0,
        ),
        client,
        expected_model_id=model_id,
        expected_model_revision=revision,
    )

    async def exercise() -> None:
        health = await process.start()
        assert health.model_id == model_id
        assert process.health == health
        assert socket_path.exists()
        await process.close()

    asyncio.run(exercise())

    assert client.closed is True
    assert not socket_path.exists()
    assert (tmp_path / "stdout.log").read_text(encoding="utf-8") == "1 1\n"


def test_hima_sidecar_restart_replaces_process_without_closing_client(
    tmp_path: Path,
) -> None:
    model_id = "SNUMPR/Protoss-a"
    revision = HIMA_PINNED_REVISIONS[model_id]
    socket_path = tmp_path / "hima.sock"
    script = tmp_path / "worker.py"
    script.write_text(
        "import pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).touch()\n"
        "print('started', flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    client = _FakeHealthClient(
        HIMALiveHealth(
            model_id=model_id,
            model_revision=revision,
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version=HIMA_PARSER_VERSION,
            vocabulary_version=HIMA_VOCABULARY_VERSION,
        )
    )
    process = HIMASidecarProcess(
        HIMASidecarSpec(
            command=(sys.executable, str(script), str(socket_path)),
            socket_path=socket_path,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            ready_timeout_seconds=2.0,
            shutdown_timeout_seconds=2.0,
        ),
        client,
        expected_model_id=model_id,
        expected_model_revision=revision,
    )

    async def exercise() -> None:
        await process.start()
        first_process = process._process
        await process.restart()
        assert client.closed is False
        assert process._process is not first_process
        assert process.health is not None
        await process.close()

    asyncio.run(exercise())

    assert client.closed is True
    assert (tmp_path / "stdout.log").read_text(encoding="utf-8") == "started\nstarted\n"


def test_hima_sidecar_serializes_concurrent_start_calls(tmp_path: Path) -> None:
    model_id = "SNUMPR/Protoss-a"
    revision = HIMA_PINNED_REVISIONS[model_id]
    socket_path = tmp_path / "hima.sock"
    script = tmp_path / "worker.py"
    script.write_text(
        "import pathlib, sys, time\npathlib.Path(sys.argv[1]).touch()\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    client = _FakeHealthClient(
        HIMALiveHealth(
            model_id=model_id,
            model_revision=revision,
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version=HIMA_PARSER_VERSION,
            vocabulary_version=HIMA_VOCABULARY_VERSION,
        )
    )
    process = HIMASidecarProcess(
        HIMASidecarSpec(
            command=(sys.executable, str(script), str(socket_path)),
            socket_path=socket_path,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            ready_timeout_seconds=2.0,
            shutdown_timeout_seconds=2.0,
        ),
        client,
        expected_model_id=model_id,
        expected_model_revision=revision,
    )

    async def exercise() -> None:
        first, second = await asyncio.gather(process.start(), process.start())
        assert first == second
        assert process.health == first
        assert socket_path.exists()
        assert (tmp_path / "hima.sock.lock").exists()
        assert client.closed is False
        await process.close()

    asyncio.run(exercise())

    assert client.closed is True
    assert not socket_path.exists()
    assert not (tmp_path / "hima.sock.lock").exists()


def test_hima_sidecar_readiness_probe_respects_the_startup_deadline(
    tmp_path: Path,
) -> None:
    model_id = "SNUMPR/Protoss-a"
    socket_path = tmp_path / "hima.sock"
    script = tmp_path / "worker.py"
    script.write_text(
        "import pathlib, sys, time\npathlib.Path(sys.argv[1]).touch()\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    health = HIMALiveHealth(
        model_id=model_id,
        model_revision=HIMA_PINNED_REVISIONS[model_id],
        adapter_version=HIMA_ADAPTER_VERSION,
        parser_version=HIMA_PARSER_VERSION,
        vocabulary_version=HIMA_VOCABULARY_VERSION,
    )
    client = _HangingHealthClient(health)
    process = HIMASidecarProcess(
        HIMASidecarSpec(
            command=(sys.executable, str(script), str(socket_path)),
            socket_path=socket_path,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            ready_timeout_seconds=0.1,
            shutdown_timeout_seconds=1.0,
        ),
        client,
        expected_model_id=model_id,
        expected_model_revision=HIMA_PINNED_REVISIONS[model_id],
    )

    with pytest.raises(HIMASidecarError, match="did not become ready within"):
        asyncio.run(process.start())

    assert client.closed is True
    assert not socket_path.exists()


def test_hima_sidecar_close_tolerates_process_exit_race(tmp_path: Path) -> None:
    model_id = "SNUMPR/Protoss-a"
    client = _FakeHealthClient(
        HIMALiveHealth(
            model_id=model_id,
            model_revision=HIMA_PINNED_REVISIONS[model_id],
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version=HIMA_PARSER_VERSION,
            vocabulary_version=HIMA_VOCABULARY_VERSION,
        )
    )
    process = HIMASidecarProcess(
        HIMASidecarSpec(
            command=(sys.executable,),
            socket_path=tmp_path / "hima.sock",
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
        ),
        client,
        expected_model_id=model_id,
        expected_model_revision=HIMA_PINNED_REVISIONS[model_id],
    )
    process._process = cast(asyncio.subprocess.Process, _VanishedProcess())

    asyncio.run(process.close())

    assert client.closed is True


def test_hima_sidecar_never_unlinks_a_preexisting_socket(tmp_path: Path) -> None:
    model_id = "SNUMPR/Protoss-a"
    socket_path = tmp_path / "hima.sock"
    socket_path.touch()
    client = _FakeHealthClient(
        HIMALiveHealth(
            model_id=model_id,
            model_revision=HIMA_PINNED_REVISIONS[model_id],
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version=HIMA_PARSER_VERSION,
            vocabulary_version=HIMA_VOCABULARY_VERSION,
        )
    )
    process = HIMASidecarProcess(
        HIMASidecarSpec(
            command=(sys.executable,),
            socket_path=socket_path,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
        ),
        client,
        expected_model_id=model_id,
        expected_model_revision=HIMA_PINNED_REVISIONS[model_id],
    )

    with pytest.raises(HIMASidecarError, match="socket already exists"):
        asyncio.run(process.start())

    assert socket_path.exists()
    assert not (tmp_path / "hima.sock.lock").exists()


def test_hima_sidecar_process_rejects_health_identity_mismatch(
    tmp_path: Path,
) -> None:
    expected_model_id = "SNUMPR/Protoss-a"
    actual_model_id = "SNUMPR/Protoss-b"
    socket_path = tmp_path / "hima.sock"
    script = tmp_path / "worker.py"
    script.write_text(
        "import pathlib, sys, time\npathlib.Path(sys.argv[1]).touch()\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    client = _FakeHealthClient(
        HIMALiveHealth(
            model_id=actual_model_id,
            model_revision=HIMA_PINNED_REVISIONS[actual_model_id],
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version=HIMA_PARSER_VERSION,
            vocabulary_version=HIMA_VOCABULARY_VERSION,
        )
    )
    process = HIMASidecarProcess(
        HIMASidecarSpec(
            command=(sys.executable, str(script), str(socket_path)),
            socket_path=socket_path,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            ready_timeout_seconds=2.0,
            shutdown_timeout_seconds=2.0,
        ),
        client,
        expected_model_id=expected_model_id,
        expected_model_revision=HIMA_PINNED_REVISIONS[expected_model_id],
    )

    with pytest.raises(HIMASidecarError, match="model mismatch"):
        asyncio.run(process.start())

    assert client.closed is True
    assert not socket_path.exists()


def test_hima_sidecar_process_rejects_stale_protocol_component(
    tmp_path: Path,
) -> None:
    model_id = "SNUMPR/Protoss-a"
    socket_path = tmp_path / "hima.sock"
    script = tmp_path / "worker.py"
    script.write_text(
        "import pathlib, sys, time\npathlib.Path(sys.argv[1]).touch()\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    client = _FakeHealthClient(
        HIMALiveHealth(
            model_id=model_id,
            model_revision=HIMA_PINNED_REVISIONS[model_id],
            adapter_version=HIMA_ADAPTER_VERSION,
            parser_version="stale-parser",
            vocabulary_version=HIMA_VOCABULARY_VERSION,
        )
    )
    process = HIMASidecarProcess(
        HIMASidecarSpec(
            command=(sys.executable, str(script), str(socket_path)),
            socket_path=socket_path,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            ready_timeout_seconds=2.0,
            shutdown_timeout_seconds=2.0,
        ),
        client,
        expected_model_id=model_id,
        expected_model_revision=HIMA_PINNED_REVISIONS[model_id],
    )

    with pytest.raises(HIMASidecarError, match="parser version mismatch"):
        asyncio.run(process.start())

    assert client.closed is True
    assert not socket_path.exists()


@pytest.mark.parametrize("race", ["terran", "zerg"])
def test_hima_sidecar_accepts_race_specific_protocol_versions(
    tmp_path: Path,
    race: str,
) -> None:
    model_id = f"SNUMPR/{race.title()}-a"
    health = HIMALiveHealth(
        model_id=model_id,
        model_revision=HIMA_PINNED_REVISIONS[model_id],
        adapter_version=HIMA_ADAPTER_VERSION,
        parser_version=HIMA_PARSER_VERSIONS[race],
        vocabulary_version=HIMA_VOCABULARY_VERSIONS[race],
    )
    process = HIMASidecarProcess(
        HIMASidecarSpec(
            command=(sys.executable, "-c", "pass"),
            socket_path=tmp_path / f"{race}.sock",
            stdout_path=tmp_path / f"{race}.stdout.log",
            stderr_path=tmp_path / f"{race}.stderr.log",
        ),
        cast(HIMALivePolicyClient, object()),
        expected_model_id=model_id,
        expected_model_revision=HIMA_PINNED_REVISIONS[model_id],
    )

    process._validate_health(health)
