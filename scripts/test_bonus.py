"""CPU checks for bonus safety, resume provenance and generated Kaggle cells."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab22 import bonus as B


@pytest.fixture
def setup_bonus(tmp_path, monkeypatch):
    from lab22 import config as C

    for name, relative in {"SFT_MERGED": "models/sft-merged", "PREF_DIR": "data/pref",
                           "EVAL_DIR": "data/eval", "VARIANTS_DIR": "adapters/variants",
                           "ADAPTERS": "adapters", "MODELS": "models", "SCREENSHOTS": "submission/screenshots",
                           "GGUF_DIR": "gguf", "DPO_ADAPTER": "adapters/dpo"}.items():
        folder = tmp_path / relative
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(C, name, folder)
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    (C.SFT_MERGED / "config.json").write_text('{}')
    (C.SFT_MERGED / "model.safetensors").write_bytes(b"test weights")
    for name in ("train", "eval"):
        (C.PREF_DIR / f"{name}.parquet").write_bytes(name.encode())
    return C


def test_config_only_zip_is_not_weights(tmp_path):
    (tmp_path / "config.json").write_text('{}')
    with pytest.raises(FileNotFoundError, match="restore the checkpoint"):
        B.require_weights(tmp_path)
    with pytest.raises(FileNotFoundError):
        B.require_weights(tmp_path, adapter=True)


def test_all_weight_shards_are_required(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "shard1.safetensors", "b": "shard2.safetensors"}}))
    (tmp_path / "shard1.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        B.require_weights(tmp_path)
    (tmp_path / "shard2.safetensors").write_bytes(b"x")
    B.require_weights(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x")
    B.require_weights(tmp_path, adapter=True)


def test_json_keeps_missing_values_null_and_atomic(tmp_path):
    path = tmp_path / "evidence.json"
    B.save_json(path, {"nan": float("nan"), "nested": [float("inf"), None, 0.7]})
    assert json.loads(path.read_text()) == {"nan": None, "nested": [None, None, 0.7]}
    assert not path.with_suffix('.json.tmp').exists()


def test_resume_requires_identical_manifest_and_complete_probes(setup_bonus):
    C = setup_bonus
    manifest = B.manifest("variants", "dpo", 300, 100, ["probe"], B.VARIANTS["dpo"])
    path = C.EVAL_DIR / "bonus-variants-dpo.json"
    assert B.cached_result(path, manifest) is None
    B.save_json(path, {"manifest": manifest, "complete": True, "outputs": []})
    assert B.cached_result(path, manifest) is None
    record = {"manifest": manifest, "complete": True, "outputs": [{"output": "x"}]}
    B.save_json(path, record)
    assert B.cached_result(path, manifest) == record
    for replacement in ({**manifest, "seed": manifest["seed"] + 1},
                        {**manifest, "n_train": 800},
                        {**manifest, "probe_sha256": "different"}):
        with pytest.raises(RuntimeError, match="different experiment"):
            B.cached_result(path, replacement)


def test_resume_rejects_changed_data_or_reference(setup_bonus):
    C = setup_bonus
    old = B.manifest("beta", "dpo-b0.10", 800, 100, ["p"], {"beta": 0.1})
    (C.PREF_DIR / "eval.parquet").write_bytes(b"reshuffled")
    assert B.manifest("beta", "dpo-b0.10", 800, 100, ["p"], {"beta": 0.1}) != old
    old = B.manifest("beta", "dpo-b0.10", 800, 100, ["p"], {"beta": 0.1})
    (C.SFT_MERGED / "model.safetensors").write_bytes(b"different trained weights")
    assert B.manifest("beta", "dpo-b0.10", 800, 100, ["p"], {"beta": 0.1}) != old


def test_variants_save_incrementally_before_later_failure(setup_bonus, monkeypatch):
    C = setup_bonus
    monkeypatch.setattr(B, "_datasets", lambda n: ([1]*n, [1]*100, ["p"]))
    calls = []

    def one(stage, name, train, held, probes, overrides):
        calls.append((stage, name, len(train), len(held)))
        if name == "dpo_norm":
            raise RuntimeError("simulated failure")
        return {"metrics": {"eval_reward_accuracy": 0.7}}

    monkeypatch.setattr(B, "_train_one", one)
    with pytest.raises(RuntimeError, match="simulated failure"):
        B.run_variants()
    saved = json.loads((C.VARIANTS_DIR / "variants_summary.json").read_text())
    assert set(saved) == {"dpo", "rpo"}
    assert all(n_train == C.TIER.variant_train and n_eval == 100 for _, _, n_train, n_eval in calls)
    assert not (C.ADAPTERS / "dpo" / "adapter_config.json").exists()


def test_beta_sweep_uses_full_split_and_all_three_betas(setup_bonus, monkeypatch):
    C = setup_bonus
    monkeypatch.setattr(B, "_datasets", lambda n: ([1]*800, [1]*100, ["p"]))
    calls = []

    def one(stage, name, train, held, probes, overrides):
        calls.append((stage, name, len(train), overrides["beta"]))
        if overrides["beta"] == 0.5:
            raise RuntimeError("stop before plotting")
        return {"metrics": {"beta": overrides["beta"]}}

    monkeypatch.setattr(B, "_train_one", one)
    with pytest.raises(RuntimeError, match="stop before plotting"):
        B.run_beta_sweep()
    assert [b for _, _, _, b in calls] == [0.05, 0.1, 0.5]
    assert all(stage == "beta" and n == 800 for stage, _, n, _ in calls)
    assert len(json.loads((C.EVAL_DIR / "bonus-beta-summary.json").read_text())) == 2


@pytest.mark.parametrize("name", ["dpo", "orpo"])
def test_training_saves_real_result_separates_weights_and_resumes(setup_bonus, monkeypatch, name):
    from lab22 import modeling as MD

    C = setup_bonus
    monkeypatch.setenv("BONUS_WEIGHTS_ROOT", str(C.REPO_ROOT / "bulky"))
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace())
    cuda = SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: None,
                           max_memory_allocated=lambda: 3 * 1024**3)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    seeds, trainer_calls, cleanup = [], [], []
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(set_seed=seeds.append))

    class Model:
        def save_pretrained(self, path):
            folder = Path(path)
            (folder / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(C.SFT_MERGED)}))
            (folder / "adapter_model.safetensors").write_bytes(b"trained adapter")

    class Tokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer_config.json").write_text('{}')

    class Trainer:
        def __init__(self, **kwargs):
            trainer_calls.append(kwargs)
            self.model = kwargs['model']
            self.state = SimpleNamespace(log_history=[
                {"step": 1, "rewards/chosen": 0.0, "rewards/rejected": 0.0},
                {"step": 2, "rewards/chosen": 0.2, "rewards/rejected": 0.1},
            ])

        def train(self):
            return SimpleNamespace(training_loss=0.65)

        def evaluate(self):
            return {"eval_rewards/chosen": 0.2, "eval_rewards/rejected": 0.1,
                    "eval_rewards/accuracies": 0.7}

    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(DPOTrainer=Trainer))
    monkeypatch.setitem(sys.modules, "trl.experimental", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "trl.experimental.orpo", SimpleNamespace(ORPOTrainer=Trainer, ORPOConfig=lambda **k: k))
    monkeypatch.setattr(MD, "load_model", lambda path: (Model(), Tokenizer()))
    monkeypatch.setattr(MD, "add_lora", lambda m: m)
    monkeypatch.setattr(MD, "dpo_config", lambda path, **kwargs: kwargs)
    monkeypatch.setattr(MD, "precision_flags", lambda: {"fp16": True})
    monkeypatch.setattr(MD, "generate", lambda *args, **kwargs: ["generated answer"])
    monkeypatch.setattr(MD, "cleanup", lambda: cleanup.append(True))
    core = C.DPO_ADAPTER / "adapter_config.json"
    core.write_text('{"core":"unchanged"}')
    record = B._train_one("variants", name, [1]*300, [1]*100, ["p"], B.VARIANTS[name])
    assert record['complete'] and record['metrics']['eval_reward_accuracy'] == 0.7
    assert record['metrics']['peak_allocated_gib_train_eval'] == 3
    assert record['metrics']['eval_reward_gap'] == pytest.approx(0.1)
    assert (Path(record['metrics']['weights_dir']) / 'adapter_model.safetensors').exists()
    assert (C.VARIANTS_DIR / name / 'adapter_config.json').exists()
    assert not (C.VARIANTS_DIR / name / 'adapter_model.safetensors').exists()
    assert (C.EVAL_DIR / f'bonus-variants-{name}.json').exists()
    assert core.read_text() == '{"core":"unchanged"}'
    assert seeds == [C.SEED] and cleanup == [True]
    assert B._train_one("variants", name, [1]*300, [1]*100, ["p"], B.VARIANTS[name]) == record
    assert len(trainer_calls) == 1


def test_bonus_notebook_is_current_append_only_and_optional_benchmark():
    from build_kaggle import BONUS_TARGET, render_bonus

    nb = render_bonus()
    assert json.loads(BONUS_TARGET.read_text(encoding="utf-8")) == nb
    cells = ["".join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    for source in cells:
        ast.parse(source.split('\n', 1)[1] if source.startswith('%%writefile ') else source)
    runtime = '\n'.join(s for s in cells if not s.startswith('%%writefile '))
    assert 'RUN_BENCHMARK = False' in runtime
    assert '/tmp/lab22-bonus-weights' in runtime
    assert 'B.run_variants()' in runtime and 'B.run_beta_sweep()' in runtime
    assert 'B.require_weights(C.SFT_MERGED)' in runtime
    assert 'lab22-bonus-evidence.zip' in runtime
    assert 'OPENROUTER_API_KEY' not in runtime
    assert not any('01_sft_mini' in s or '03_dpo_train' in s for s in cells)
    installer = next(s for s in cells if "'pip', 'install'" in s)
    assert installer.startswith('if RUN_BENCHMARK:')
    assert 'lm-eval' in installer and 'llama-cpp' not in installer
