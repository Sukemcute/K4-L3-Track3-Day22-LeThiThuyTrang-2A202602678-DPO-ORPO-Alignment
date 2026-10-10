"""Submission checks must stay strict when importing cloud evidence on Windows."""
import json

import pytest

import verify as V
from lab22.data import save_split_fingerprint


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "REPO", tmp_path)
    adapter = tmp_path / "adapters" / "dpo"
    adapter.mkdir(parents=True)
    merged = tmp_path / "models" / "sft-merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text('{}', encoding="utf-8")
    pref = tmp_path / "data" / "pref"
    pref.mkdir(parents=True)
    for name in ("train", "eval"):
        (pref / f"{name}.parquet").write_bytes(name.encode())
    save_split_fingerprint(pref, adapter)
    metrics = {"reference": "models/sft-merged (precomputed)", "end_reward_gap": 0.1,
               "eval_reward_accuracy": 0.7, "diagnosis": "INTENDED"}
    (adapter / "dpo_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return tmp_path, adapter


def run_dpo(adapter, base):
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}), encoding="utf-8")
    problems, warnings = [], []
    V.check_dpo(problems, warnings)
    return problems, warnings


@pytest.mark.parametrize("base", sorted(V.EXPORTED_SFT_REFERENCES))
def test_cloud_reference_preserves_config_and_checks_evidence(evidence, base):
    root, adapter = evidence
    problems, warnings = run_dpo(adapter, base)
    assert not problems
    assert any("do not verify or restore model weights" in w for w in warnings)
    assert json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"] == base


@pytest.mark.parametrize("base", ["", "unsloth/Qwen3-4B", "/other/models/sft-merged",
                                  "/kaggle/working/wrong/models/sft-merged",
                                  "/kaggle/working/lab22/models/../models/sft-merged"])
def test_wrong_reference_is_still_rejected(evidence, base):
    _, adapter = evidence
    problems, _ = run_dpo(adapter, base)
    assert any("WRONG REF" in p for p in problems)


def test_relative_reference_is_repo_relative_not_cwd(evidence, monkeypatch, tmp_path):
    _, adapter = evidence
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert run_dpo(adapter, "models/sft-merged") == ([], [])


def test_absolute_local_reference(evidence):
    root, adapter = evidence
    assert run_dpo(adapter, str(root / "models" / "sft-merged")) == ([], [])


@pytest.mark.parametrize("missing", ["config", "reference", "metrics", "split", "changed_split"])
def test_cloud_reference_does_not_bypass_missing_or_changed_evidence(evidence, missing):
    root, adapter = evidence
    if missing == "config":
        (root / "models" / "sft-merged" / "config.json").unlink()
    elif missing == "reference":
        (adapter / "dpo_metrics.json").write_text('{}', encoding="utf-8")
    elif missing == "metrics":
        (adapter / "dpo_metrics.json").unlink()
    elif missing == "split":
        (adapter / "split.json").unlink()
    else:
        (root / "data" / "pref" / "eval.parquet").write_bytes(b"changed")
    problems, _ = run_dpo(adapter, "/kaggle/working/lab22/models/sft-merged")
    assert problems
