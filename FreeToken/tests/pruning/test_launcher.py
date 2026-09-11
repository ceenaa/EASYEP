import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from freetoken.pruning.artifacts import all_ones_mask, checkpoint_identity, write_new
from test_artifacts import checkpoint


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[3] / "launcher/start_pruning_server.py"
    spec = importlib.util.spec_from_file_location("pruning_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(checkpoint, **overrides):
    return SimpleNamespace(**{**dict(model_dir=checkpoint, phase="baseline", artifact=None,
        max_seq_len=8192, chunk_tokens=256, cache_slots=512, port=1919), **overrides})


def test_launch_modes_validate_artifacts_before_loading(launcher, checkpoint, tmp_path):
    command = launcher.build_command(args(checkpoint))
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--cuda-graph-max-bs") + 1] == "0"
    assert command[command.index("--cache-type") + 1] == "naive"
    mask = tmp_path / "mask.json"
    write_new(mask, all_ones_mask(checkpoint_identity(checkpoint)))
    command = launcher.build_command(args(checkpoint, phase="mask", artifact=mask))
    assert command[-2:] == ["--expert-mask", str(mask)]
    with pytest.raises(FileExistsError):
        launcher.build_command(args(checkpoint, phase="statistics", artifact=mask))
    with pytest.raises(ValueError):
        launcher.build_command(args(checkpoint, phase="mask"))
    with pytest.raises(ValueError):
        launcher.build_command(args(checkpoint, max_seq_len=8191))


def test_slurm_mig_keeps_cuda_visibility_and_live_tracing(launcher, checkpoint, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-assigned-by-slurm")
    command = launcher.build_command(args(checkpoint, phase="statistics",
        artifact=tmp_path / "statistics.json", trace_decode=True, inherit_gpu=True))
    assert "--gpu" not in command
    assert "--expert-stats-decode" in command
    assert command[command.index("--moe-strategy") + 1] == "offload"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "MIG-assigned-by-slurm"
    default = launcher.build_command(args(checkpoint))
    assert default[default.index("--gpu") + 1] == "0"
