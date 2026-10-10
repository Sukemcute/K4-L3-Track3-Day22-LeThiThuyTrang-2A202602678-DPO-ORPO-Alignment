"""Resumable bonus experiments. GPU libraries are imported only when running.

Evidence stays in the repo; optional BONUS_WEIGHTS_ROOT puts bulky adapters in
/tmp on Kaggle. Never overwrite the core SFT/DPO artifacts or judge results.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import config as C
from . import data as D

VARIANTS = {
    "dpo": {"loss_type": ["sigmoid"]},
    "rpo": {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 1.0]},
    "dpo_norm": {"loss_type": ["sigmoid_norm"]},
    "ld_dpo": {"loss_type": ["sigmoid"], "ld_alpha": 0.5},
    "orpo": {"beta": 0.1},
}


def require_weights(folder: Path, adapter: bool = False) -> None:
    """A config-only evidence ZIP is not a runnable model checkpoint."""
    if adapter:
        present = any((folder / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin"))
    else:
        indexes = list(folder.glob("*.index.json"))
        if indexes:
            shards = set()
            for index in indexes:
                shards.update(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
            present = bool(shards) and all((folder / n).is_file() for n in shards)
        else:
            present = any((folder / n).is_file() for n in ("model.safetensors", "pytorch_model.bin"))
    if not present:
        raise FileNotFoundError(f"Missing {'adapter' if adapter else 'model'} weights in {folder}. Evidence ZIP contains configs, not weights; restore the checkpoint, do not use the raw base instead.")


def clean_json(value):
    """Strict JSON: absent/non-finite metrics remain null, never fabricated zeros."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def manifest(stage: str, name: str, n_train: int, n_eval: int, probes: list[str], overrides: dict) -> dict:
    config = C.SFT_MERGED / "config.json"
    weights = sorted([*C.SFT_MERGED.glob("*.safetensors"), *C.SFT_MERGED.glob("pytorch_model*.bin")])
    runtime = {}
    for package in ("unsloth", "trl", "transformers", "peft", "torch", "bitsandbytes", "datasets"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = None
    return {
        "schema_version": 1, "stage": stage, "name": name,
        "reference": str(C.SFT_MERGED.resolve()), "split_sha256": D.split_fingerprint(C.PREF_DIR),
        "reference_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "reference_weight_stats": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in weights},
        "reference_tokenizer_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                       (C.SFT_MERGED / "tokenizer_config.json", C.SFT_MERGED / "tokenizer.json") if p.is_file()},
        "runtime_versions": runtime,
        "implementation_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                  for name in ("bonus.py", "modeling.py", "data.py")},
        "n_train": n_train, "n_eval": n_eval, "seed": C.SEED,
        "epochs": C.DPO_EPOCHS, "lr": C.DPO_LR, "beta": C.DPO_BETA,
        "max_length": C.MAX_LEN, "lora_r": C.LORA_R, "lora_alpha": C.LORA_ALPHA,
        "lora_targets": C.LORA_TARGETS, "compute_tier": C.COMPUTE_TIER,
        "batch": C.TIER.dpo_batch, "grad_accum": C.TIER.dpo_grad_accum,
        "probe_sha256": hashlib.sha256(json.dumps(probes, ensure_ascii=False).encode()).hexdigest(),
        "probe_count": len(probes), "generation_max_new_tokens": 256,
        "chat_template_kwargs": C.CHAT_TEMPLATE_KWARGS, "overrides": overrides,
    }


def cached_result(path: Path, expected: dict) -> dict | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("manifest") != expected:
        raise RuntimeError(f"{path} belongs to a different experiment. Preserve it and choose a new output location; do not mix splits/settings.")
    if record.get("complete") and len(record.get("outputs", [])) == expected["probe_count"]:
        return record
    return None


def _datasets(n_train: int | None):
    require_weights(C.SFT_MERGED)
    mismatch = D.split_mismatch(C.PREF_DIR, C.DPO_ADAPTER)
    if mismatch:
        raise RuntimeError(f"Core DPO/data mismatch: {mismatch}")
    from datasets import Dataset

    train = Dataset.from_parquet(str(C.PREF_DIR / "train.parquet"))
    held = Dataset.from_parquet(str(C.PREF_DIR / "eval.parquet"))
    if not len(train) or not len(held):
        raise ValueError("Empty preference split")
    D.assert_disjoint(list(train), list(held))
    if n_train is not None:
        train = train.select(range(min(n_train, len(train))))
    probes = [r["prompt"][0]["content"] for r in held.select(range(min(20, len(held))))]
    return train, held, probes


def _train_one(stage: str, name: str, train, held, probes: list[str], overrides: dict) -> dict:
    expected = manifest(stage, name, len(train), len(held), probes, overrides)
    record_path = C.EVAL_DIR / f"bonus-{stage}-{name}.json"
    cached = cached_result(record_path, expected)
    if cached is not None:
        print(f"Reuse completed {stage}/{name}: identical split, settings and probes")
        return cached

    import unsloth  # noqa: F401
    import torch
    from transformers import set_seed
    from . import modeling as MD

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU for bonus training")
    set_seed(C.SEED)
    evidence_dir = C.VARIANTS_DIR / name if stage == "variants" else C.ADAPTERS / name
    weights_root = os.environ.get("BONUS_WEIGHTS_ROOT")
    weights_dir = Path(weights_root) / stage / name if weights_root else evidence_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    model = trainer = None
    try:
        model, tokenizer = MD.load_model(C.SFT_MERGED)
        model = MD.add_lora(model)
        torch.cuda.reset_peak_memory_stats()
        if name == "orpo":
            from trl.experimental.orpo import ORPOConfig, ORPOTrainer

            args = ORPOConfig(
                output_dir=str(weights_dir / "checkpoints"),
                per_device_train_batch_size=C.TIER.dpo_batch,
                per_device_eval_batch_size=C.TIER.dpo_batch,
                gradient_accumulation_steps=C.TIER.dpo_grad_accum,
                num_train_epochs=C.DPO_EPOCHS, learning_rate=C.DPO_LR,
                beta=0.1, max_length=C.MAX_LEN, warmup_steps=0.1,
                lr_scheduler_type="cosine", logging_steps=5, eval_strategy="no",
                save_strategy="no", optim="adamw_8bit", seed=C.SEED,
                report_to="none", **MD.precision_flags(),
            )
            trainer = ORPOTrainer(model=model, args=args, train_dataset=train, eval_dataset=held, processing_class=tokenizer)
        else:
            from trl import DPOTrainer

            args = MD.dpo_config(weights_dir / "checkpoints", eval_strategy="no", **overrides)
            trainer = DPOTrainer(model=model, ref_model=None, args=args,
                                 train_dataset=train, eval_dataset=held, processing_class=tokenizer)
        started = time.monotonic()
        trained = trainer.train()
        elapsed = time.monotonic() - started
        ev = trainer.evaluate()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        history = list(trainer.state.log_history)
        # Save before generation; MD.generate switches the model to inference mode.
        trainer.model.save_pretrained(str(weights_dir))
        tokenizer.save_pretrained(str(weights_dir))
        D.save_split_fingerprint(C.PREF_DIR, weights_dir)
        outputs = MD.generate(trainer.model, tokenizer, probes, max_new_tokens=256, batch_size=1)
        train_hist = MD.reward_history(history)
        ev_hist = MD.reward_history(history, prefix="eval_")
        diagnosis = MD.diagnose(ev_hist if len(ev_hist) >= 2 else train_hist)[0] if name != "orpo" else "NOT COMPARABLE (ORPO)"
        chosen, rejected = ev.get("eval_rewards/chosen"), ev.get("eval_rewards/rejected")
        metrics = {
            "beta": overrides.get("beta", 0.1 if name == "orpo" else C.DPO_BETA),
            "eval_reward_accuracy": ev.get("eval_rewards/accuracies"),
            "eval_chosen_reward": chosen, "eval_rejected_reward": rejected,
            "eval_reward_gap": chosen - rejected if chosen is not None and rejected is not None else None,
            "eval_log_odds_ratio": ev.get("eval_log_odds_ratio"),
            "mean_output_chars": sum(map(len, outputs)) / len(outputs),
            "final_train_loss": float(trained.training_loss), "diagnosis": diagnosis,
            "training_seconds": elapsed, "peak_allocated_gib_train_eval": peak,
            "weights_dir": str(weights_dir), "n_train": len(train), "n_eval": len(held),
        }
        record = {"manifest": expected, "metrics": metrics, "history": history,
                  "outputs": [{"prompt": p, "output": o} for p, o in zip(probes, outputs)], "complete": True}
        # Mirror only small provenance files when weights are stored outside working.
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("adapter_config.json", "split.json"):
            content = json.loads((weights_dir / filename).read_text(encoding="utf-8"))
            save_json(evidence_dir / filename, content)
        save_json(evidence_dir / "experiment.json", expected)
        save_json(evidence_dir / "dpo_metrics.json", metrics)
        save_json(record_path, record)
        print(json.dumps(clean_json(metrics), ensure_ascii=False, indent=2))
        return clean_json(record)
    finally:
        del trainer, model
        MD.cleanup()


def _plot(records: dict[str, dict], filename: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.DataFrame({k: v["metrics"] for k, v in records.items()}).T
    print(table.to_string())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    table["eval_reward_accuracy"].astype(float).plot.bar(ax=axes[0], color="#2e548a")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("held-out pair reward accuracy (not judge win rate)")
    table["mean_output_chars"].astype(float).plot.bar(ax=axes[1], color="#c83538")
    axes[1].set_title("mean output length: same held-out probes")
    fig.tight_layout()
    C.SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.SCREENSHOTS / filename, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_variants() -> dict:
    C.ensure_dirs()
    train, held, probes = _datasets(C.TIER.variant_train)
    records = {}
    for name, overrides in VARIANTS.items():
        records[name] = _train_one("variants", name, train, held, probes, overrides)
        save_json(C.VARIANTS_DIR / "variants_summary.json", {k: v["metrics"] for k, v in records.items()})
    _plot(records, "03b-variants.png")
    return {k: v["metrics"] for k, v in records.items()}


def run_beta_sweep() -> list[dict]:
    C.ensure_dirs()
    train, held, probes = _datasets(None)
    records = {}
    for beta in (0.05, 0.1, 0.5):
        name = f"dpo-b{beta:.2f}"
        records[name] = _train_one("beta", name, train, held, probes, {"loss_type": ["sigmoid"], "beta": beta})
        save_json(C.EVAL_DIR / "bonus-beta-summary.json", {k: v["metrics"] for k, v in records.items()})
    # Plot uses beta-scaled rewards; the caption explicitly warns about this.
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("lab22_sweep_plot", C.REPO_ROOT / "scripts" / "eval_judge.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [r["metrics"] for r in records.values()]
    module.plot_sweep(rows, C.SCREENSHOTS / "bonus-beta-sweep.png")
    return rows
