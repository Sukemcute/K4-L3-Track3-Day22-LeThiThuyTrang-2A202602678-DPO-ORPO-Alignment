# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB6 — Đánh giá chuẩn (benchmark) SFT vs SFT+DPO bằng lm-eval (TUỲ CHỌN, thưởng điểm)
#
# **Công nghệ:** `lm-eval` 0.4.13 trên mô hình 16-bit: `models/sft-merged` (SFT) và
# `models/sft-merged` + `peft=adapters/dpo` (SFT+DPO).
#
# Ba lỗi của bản cũ đã sửa:
# 1. **Chat template.** Mô hình chat bị chấm như mô hình base (không template) thì điểm
#    IFEval/GSM8K không phản ánh cách nó được dùng. Ở đây truyền `--apply_chat_template`
#    và `--fewshot_as_multiturn`.
# 2. **`--limit` tính theo từng subtask.** `--limit 500` trên nhóm `mmlu` (57 môn) là
#    ~28k câu, không phải 500. Limit MMLU ở đây được đặt *theo môn*.
# 3. **AlpacaEval-lite bị bỏ.** `tatsu-lab/alpaca_eval` là bộ dữ liệu dạng script, không
#    nạp được với `datasets` ≥ 4. Đánh giá theo kiểu giám khảo đã có ở NB4.
#
# Thêm `global_mmlu_full_vi` (MMLU dịch tiếng Việt, có kiểm duyệt) vì lab là tiếng Việt.
#
# > Điểm có thể **giảm** sau DPO (alignment tax, tức thuế căn chỉnh). Với limit nhỏ, chênh lệch ±2–3 điểm
# > thường nằm trong nhiễu: xem cột `stderr` trước khi kết luận.

# %%
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

import torch

from lab22 import config as C
from lab22.bonus import require_weights

assert torch.cuda.is_available()
assert C.SFT_MERGED.exists() and C.DPO_ADAPTER.exists(), "Run NB1 + NB3 first"
require_weights(C.SFT_MERGED)
require_weights(C.DPO_ADAPTER, adapter=True)
C.ensure_dirs()

BIG = C.COMPUTE_TIER == "BIGGPU"
# name: (task, num_fewshot, limit per subtask or None for all, primary metric)
BENCHMARKS = {
    "IFEval": ("ifeval", 0, None if BIG else 200, "prompt_level_strict_acc,none"),
    "GSM8K": ("gsm8k", 5, None if BIG else 250, "exact_match,flexible-extract"),
    "Global-MMLU-vi": ("global_mmlu_full_vi", 0, 40 if BIG else 10, "acc,none"),
}
DTYPE = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
BATCH = os.environ.get("BENCH_BATCH", "auto" if BIG else "1")
for name, (task, shots, limit, _metric) in BENCHMARKS.items():
    print(f"{name:15s} task={task} fewshot={shots} limit/subtask={limit or 'all'}")

# %% [markdown]
# ## 1. Hàm chạy lm-eval

# %%
def run_lm_eval(label: str, task: str, shots: int, limit: int | None) -> dict:
    model_args = f"pretrained={C.SFT_MERGED},dtype={DTYPE},enable_thinking=False"
    if label == "dpo":
        model_args += f",peft={C.DPO_ADAPTER}"
    out_dir = C.EVAL_DIR / "lm_eval" / f"{label}-{task}"
    cmd = [
        "lm_eval", "--model", "hf", "--model_args", model_args,
        "--tasks", task, "--num_fewshot", str(shots),
        "--apply_chat_template", "--batch_size", BATCH,
        "--device", "cuda:0", "--seed", str(C.SEED), "--output_path", str(out_dir),
    ]
    if shots:
        cmd.append("--fewshot_as_multiturn")
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"\n>>> {label} · {task}\n{' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    files = sorted(out_dir.glob("**/results*.json"), key=lambda p: p.stat().st_mtime)
    if proc.returncode != 0 or not files:
        print(proc.stderr[-2000:])
        raise RuntimeError(f"lm_eval failed for {label}/{task}")
    return json.loads(files[-1].read_text())


def score(result: dict, task: str, metric: str) -> tuple[float, float]:
    # Groups (global_mmlu_full_vi) report the aggregate under the group name.
    block = result.get("groups", {}).get(task) or result["results"][task]
    stderr_key = metric.replace(",", "_stderr,", 1)
    return float(block[metric]), float(block.get(stderr_key, float("nan")))


# %% [markdown]
# ## 2. Chạy (T4: ~40–60 phút cho cả hai điều kiện)

# %%
rows = []
for name, (task, shots, limit, metric) in BENCHMARKS.items():
    row = {"benchmark": name, "task": task, "limit_per_subtask": limit}
    for label in ("sft", "dpo"):
        value, err = score(run_lm_eval(label, task, shots, limit), task, metric)
        row[label], row[f"{label}_stderr"] = value, err
    row["delta"] = row["dpo"] - row["sft"]
    rows.append(row)
    (C.EVAL_DIR / "benchmark_progress.json").write_text(
        json.dumps({"complete": len(rows) == len(BENCHMARKS), "results": rows}, indent=2)
    )
    print(row)

# %% [markdown]
# ## 3. Biểu đồ (sản phẩm nộp `07-benchmark-comparison.png`)

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.DataFrame(rows)
print(df.to_string(index=False))

x = np.arange(len(df))
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x - 0.18, df["sft"], 0.36, yerr=df["sft_stderr"], label="SFT", color="#2e548a", capsize=3)
ax.bar(x + 0.18, df["dpo"], 0.36, yerr=df["dpo_stderr"], label="SFT+DPO", color="#c83538", capsize=3)
for i, r in df.iterrows():
    ax.annotate(f"Δ={r['delta']:+.3f}", (x[i], max(r["sft"], r["dpo"]) + 0.05), ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(df["benchmark"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("score (± stderr)")
ax.set_title(f"lm-eval · chat template · {C.COMPUTE_TIER}")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.savefig(C.SCREENSHOTS / "07-benchmark-comparison.png", dpi=120, bbox_inches="tight")
plt.show()

# %%
(C.EVAL_DIR / "benchmark_results.json").write_text(
    json.dumps({"compute_tier": C.COMPUTE_TIER, "dtype": DTYPE, "chat_template": True, "results": rows}, indent=2)
)
print("Saved data/eval/benchmark_results.json")

# %% [markdown]
# ## 4. Đọc kết quả (REFLECTION §7)
#
# 1. |Δ| có lớn hơn ~2× stderr không? Nếu không, đừng gọi là "tăng" hay "giảm".
# 2. IFEval đo khả năng làm đúng định dạng: DPO trên dữ liệu chat thường giúp hoặc giữ nguyên.
# 3. GSM8K giảm ⇒ alignment tax (thuế căn chỉnh); NB7 (GRPO với reward kiểm chứng được) là một cách lấy lại.
# 4. Global-MMLU-vi gần như phẳng là bình thường: DPO không dạy kiến thức mới.
#
# **Tiếp theo:** NB7 (GRPO, thưởng điểm) hoặc `make verify`.
