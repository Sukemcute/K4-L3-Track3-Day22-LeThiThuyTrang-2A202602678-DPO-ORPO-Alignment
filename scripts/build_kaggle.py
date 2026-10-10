#!/usr/bin/env python3
"""Generate Kaggle notebooks: core, API rejudge, and live-session bonuses.

    python scripts/build_kaggle.py
    python scripts/build_kaggle.py --check
"""
from __future__ import annotations

import argparse
import json

from build_colab import RELEASE_GPU, REPO, STAGES, code, md, percent_cells, requirements

TARGET = REPO / "kaggle" / "Lab22_DPO_T4_Kaggle.ipynb"
REJUDGE_TARGET = REPO / "kaggle" / "Lab22_OpenRouter_Rejudge.ipynb"
BONUS_TARGET = REPO / "kaggle" / "Lab22_Bonus_T4_Kaggle.ipynb"
WORKDIR = "/kaggle/working/lab22"
OPENROUTER_CONFIG = (
    "import os\n"
    "from kaggle_secrets import UserSecretsClient\n"
    'os.environ["JUDGE_PROVIDER"] = "openrouter"\n'
    'os.environ["JUDGE_MODEL"] = "google/gemini-2.5-flash"\n'
    "try:\n"
    '    _judge_key = UserSecretsClient().get_secret("OPENROUTER_API_KEY")\n'
    "except Exception:\n"
    "    raise RuntimeError('Add OPENROUTER_API_KEY in Kaggle Secrets and enable notebook access.') from None\n"
    "if not isinstance(_judge_key, str) or not _judge_key.strip():\n"
    "    raise RuntimeError('OPENROUTER_API_KEY in Kaggle Secrets is empty.')\n"
    'os.environ["OPENROUTER_API_KEY"] = _judge_key.strip()\n'
    "del _judge_key\n"
    'print("Judge: openrouter /", os.environ["JUDGE_MODEL"])'
)


def render() -> dict:
    # GGUF and lm-eval dependencies are used only by bonus notebooks.
    specs = [spec for spec in requirements() if not spec.startswith(("llama-cpp-python", "lm-eval"))]
    cells = [
        md(
            "# Lab 22 — DPO Alignment · Kaggle T4\n\n"
            "Sinh từ `notebooks/*.py` và `lab22/*.py` bằng `scripts/build_kaggle.py`.\n\n"
            "1. Trong Settings, chọn **GPU T4 ×2** và bật **Internet**.\n"
            "2. Thêm secret **OPENROUTER_API_KEY** trong Kaggle Secrets và bật quyền dùng cho notebook.\n"
            "3. Chạy notebook trong một phiên mới. Cấu hình dùng một GPU, tier T4.\n"
            "4. **Run All** chạy Setup → NB0 → NB1 → NB2 → NB3 → NB4; không chạy bonus.\n"
            "5. NB4 dùng **OpenRouter / google/gemini-2.5-flash**, đảo A/B hai lần mỗi cặp. "
            "Thiếu key sẽ dừng; không chuyển sang reward model local.\n"
            "6. Cuối notebook tạo `/kaggle/working/lab22-evidence.zip`. Tải ZIP và notebook "
            "có output về máy. ZIP chứa bằng chứng nhỏ, không chứa trọng số để tiếp tục training.\n\n"
            "Kết quả được ghi vào `/kaggle/working/lab22`. Mỗi giai đoạn nạp lại artifact "
            "cần thiết từ thư mục này; nếu phiên mất dữ liệu, cần chạy lại từ NB1. "
            "Cache tải model/dataset nằm trong `/tmp/lab22-hf-cache`, ngoài ổ working. "
            "Cache này chỉ dùng trong phiên hiện tại và không được đóng gói vào ZIP."
        ),
        md("## A. Setup"),
        code(
            "import os\n"
            "import sys\n"
            "if 'torch' in sys.modules or 'unsloth' in sys.modules:\n"
            "    raise RuntimeError('Restart Session, then Run All from the first cell.')\n"
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
            'os.environ["COMPUTE_TIER"] = "T4"\n'
            'os.environ["HF_HOME"] = "/tmp/lab22-hf-cache"\n'
            'os.environ["HF_HUB_CACHE"] = "/tmp/lab22-hf-cache/hub"\n'
            'os.environ["HF_DATASETS_CACHE"] = "/tmp/lab22-hf-cache/datasets"\n'
            'os.environ["HF_XET_CACHE"] = "/tmp/lab22-hf-cache/xet"\n'
            "# Read the selected judge's API key without printing it.\n"
            + OPENROUTER_CONFIG
        ),
        code(
            "import subprocess\n"
            f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *{specs!r}])"
        ),
        code(
            "from pathlib import Path\n"
            f'WORK = Path("{WORKDIR}")\n'
            '(WORK / "lab22").mkdir(parents=True, exist_ok=True)\n'
            "os.chdir(WORK)\n"
            "import torch\n"
            "assert torch.cuda.is_available(), 'Enable GPU T4 x2 in Settings, then restart the session.'\n"
            "assert torch.cuda.device_count() == 1, 'Restart the session to apply CUDA_VISIBLE_DEVICES.'\n"
            "print(f'Work directory: {WORK}')\n"
            "print(f'GPU: {torch.cuda.get_device_name(0)}; "
            "VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB')"
        ),
        md("### Helper package `lab22/`"),
    ]
    for module in sorted((REPO / "lab22").glob("*.py")):
        body = module.read_text(encoding="utf-8")
        cells.append(code(f"%%writefile {WORKDIR}/lab22/{module.name}\n{body}"))
    stages = [(stem, kind) for stem, kind in STAGES if kind.startswith("core")]
    for index, (stem, kind) in enumerate(stages):
        if index:
            cells.append(code(RELEASE_GPU))
        cells.append(md(f"---\n# ⏵ `notebooks/{stem}.py` ({kind})"))
        cells.extend(percent_cells(REPO / "notebooks" / f"{stem}.py"))
    cells.extend([
        md(
            "## B. Tải bằng chứng nộp bài\n\n"
            "Cell dưới tạo ZIP chứa screenshots, dữ liệu preference/evaluation và các JSON "
            "cấu hình/metrics. ZIP không chứa trọng số model/adapter. Tải thêm notebook này "
            "với output đã chạy; sau đó điền `submission/REFLECTION.md` bằng số liệu thật. "
            "Muốn tiếp tục training ở phiên khác, cần sao lưu riêng trọng số."
        ),
        code(
            "from zipfile import ZipFile, ZIP_DEFLATED\n"
            "evidence = set()\n"
            "for folder, suffixes in (\n"
            "    ('submission/screenshots', {'.png', '.jpg', '.jpeg'}),\n"
            "    ('data/pref', {'.parquet'}),\n"
            "    ('data/eval', {'.json', '.jsonl'}),\n"
            "    ('adapters', {'.json'}),\n"
            "):\n"
            "    evidence.update(p for p in (WORK / folder).rglob('*') if p.is_file() and p.suffix in suffixes)\n"
            "sft_config = WORK / 'models/sft-merged/config.json'\n"
            "if sft_config.is_file():\n"
            "    evidence.add(sft_config)\n"
            "archive = Path('/kaggle/working/lab22-evidence.zip')\n"
            "with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:\n"
            "    for path in sorted(evidence):\n"
            "        bundle.write(path, arcname=str(path.relative_to(WORK)))\n"
            "print(f'Saved {archive} ({len(evidence)} evidence files)')\n"
            "print('Download the ZIP and the executed notebook, then fill in REFLECTION.md.')"
        ),
    ])
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def render_rejudge() -> dict:
    """Judge existing answers without loading policy weights or requiring a GPU."""
    cells = [
        md(
            "# Lab 22 — Chấm lại NB4 bằng OpenRouter\n\n"
            "Dùng trong phiên Kaggle còn `/kaggle/working/lab22/data/eval/side_by_side.jsonl`. "
            "Nếu phiên mới, giải nén `lab22-evidence.zip` vào `/kaggle/working/lab22` trước. "
            "Không cần chạy lại NB0–NB3 hoặc sinh lại câu trả lời; không cần GPU.\n\n"
            "Thêm secret **OPENROUTER_API_KEY** trong Kaggle Secrets và bật quyền dùng cho notebook, "
            "bật **Internet**, rồi chạy các cell dưới. Có thể chép các cell này vào cuối notebook "
            "đang chạy để giữ nguyên phiên. Giám khảo: **google/gemini-2.5-flash** qua OpenRouter. "
            "58 cặp sẽ cần ít nhất 116 lượt gọi API vì đảo A/B; các lượt sai định dạng được thử lại.\n\n"
            "Kết quả RM đã lưu được giữ để đối chiếu `cross_judge`; kết quả API mới sẽ được ghi vào "
            "`judge_results_api.json` và `judge_summary.json`. Cuối notebook tạo lại ZIP bằng chứng."
        ),
        code(OPENROUTER_CONFIG),
        code(
            "import subprocess\n"
            "import sys\n"
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'openai>=1.55,<4.0'])"
        ),
        code(
            "from pathlib import Path\n"
            f'WORK = Path("{WORKDIR}")\n'
            "outputs_file = WORK / 'data/eval/side_by_side.jsonl'\n"
            "if not outputs_file.is_file():\n"
            "    raise FileNotFoundError('Restore lab22-evidence.zip into /kaggle/working/lab22 first.')\n"
            "(WORK / 'lab22').mkdir(parents=True, exist_ok=True)\n"
            "os.chdir(WORK)"
        ),
    ]
    for name in ("__init__.py", "config.py", "judge.py"):
        body = (REPO / "lab22" / name).read_text(encoding="utf-8")
        cells.append(code(f"%%writefile {WORKDIR}/lab22/{name}\n{body}"))
    cells.append(code(
        "import hashlib\n"
        "import importlib\n"
        "import json\n"
        "sys.path.insert(0, str(WORK))\n"
        "from lab22 import config as C, judge as J\n"
        "# Reload to apply OpenRouter even if C/J were imported during the earlier RM run.\n"
        "importlib.reload(C)\n"
        "importlib.reload(J)\n"
        "output_bytes = outputs_file.read_bytes()\n"
        "records = [json.loads(line) for line in output_bytes.decode('utf-8').splitlines() if line.strip()]\n"
        "assert records and all({'id', 'category', 'prompt', 'sft', 'dpo'} <= r.keys() for r in records)\n"
        "assert len({r['id'] for r in records}) == len(records), 'Duplicate output IDs'\n"
        "assert sum(r['category'] == 'heldout' for r in records) >= 50, 'Need at least 50 held-out outputs'\n"
        "OUTPUTS_SHA = hashlib.sha256(output_bytes).hexdigest()\n"
        "print(f'Loaded {len(records)} saved pairs; at least {2 * len(records)} API calls.')"
    ))
    nb4 = percent_cells(REPO / "notebooks" / "04_compare_and_eval.py")
    for cell in nb4:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith(("provider = C.JUDGE_PROVIDER", "def splits(rows:")):
            cells.append(cell)
    cells.append(md("## Tải kết quả mới\n\nTải ZIP bên dưới và notebook có output để cập nhật bài nộp."))
    cells.append(render()["cells"][-1])
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def render_bonus() -> dict:
    """Append to the LIVE core notebook: no restart, pip upgrade or core rerun."""
    cells = [
        md(
            "# Lab 22 — Bonus tiếp tục trong phiên Kaggle cũ\n\n"
            "**Không Restart Session.** Copy các cell này vào cuối notebook core còn trọng số. "
            "Import thành notebook Kaggle mới không tự mang theo thư mục của phiên cũ.\n\n"
            "Mặc định: NB3b 5 biến thể (+8) và β-sweep 3 giá trị (+6). Cộng bằng chứng chấm chéo "
            "đã có (+4) là 18 điểm tiềm năng, không phải điểm đã được chấm. NB6 (+6) tắt mặc định; "
            "bật sau nếu muốn hướng tới trần 20 bonus. Không tự upload HF, không gọi API, "
            "không chạy GGUF/GRPO trong notebook này.\n\n"
            "Có **8 lượt fine-tune mới**: 5 × 300 cặp và 3 × 800 cặp, đều 1 epoch. "
            "Không lấy baseline 800 cặp để so trực tiếp với biến thể 300 cặp. "
            "Kết quả từng run được lưu ngay để tiếp tục sau lỗi; thời gian thực tế phụ thuộc phiên. "
            "Adapter mới nằm trong `/tmp/lab22-bonus-weights` để tránh đầy working. "
            "Chúng sẽ mất khi phiên kết thúc; sao lưu riêng nếu cần dùng lại."
        ),
        code(
            "RUN_VARIANTS = True\n"
            "RUN_BETA_SWEEP = True\n"
            "RUN_BENCHMARK = False  # Optional; change to True only when ready for extra GPU work.\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            f"WORK = Path({WORKDIR!r})\n"
            "if not (WORK / 'models/sft-merged/config.json').is_file():\n"
            "    raise FileNotFoundError('Append these cells to the OLD live Kaggle notebook; do not start an empty session.')\n"
            "os.environ['COMPUTE_TIER'] = 'T4'\n"
            "os.environ['BONUS_WEIGHTS_ROOT'] = '/tmp/lab22-bonus-weights'\n"
            "# Existing imported HF constants are unchanged; child processes use /tmp cache.\n"
            "os.environ['HF_HOME'] = '/tmp/lab22-hf-cache'\n"
            "os.environ['HF_HUB_CACHE'] = '/tmp/lab22-hf-cache/hub'\n"
            "os.environ['HF_DATASETS_CACHE'] = '/tmp/lab22-hf-cache/datasets'\n"
            "os.environ['HF_XET_CACHE'] = '/tmp/lab22-hf-cache/xet'\n"
            "for folder in ('lab22', 'scripts', 'notebooks'):\n"
            "    (WORK / folder).mkdir(parents=True, exist_ok=True)\n"
            "os.chdir(WORK)"
        ),
        code(RELEASE_GPU),
    ]
    # Use the same helper source as the core notebook, but don't retrain the core.
    for module in sorted((REPO / "lab22").glob("*.py")):
        cells.append(code(f"%%writefile {WORKDIR}/lab22/{module.name}\n" + module.read_text(encoding="utf-8")))
    for path in (REPO / "scripts/eval_judge.py", REPO / "notebooks/06_benchmark.py"):
        cells.append(code(f"%%writefile {WORKDIR}/{path.relative_to(REPO).as_posix()}\n" + path.read_text(encoding="utf-8")))
    cells.append(code(
        "import importlib\n"
        "import shutil\n"
        "import unsloth\n"
        "import torch\n"
        "sys.path.insert(0, str(WORK))\n"
        "from lab22 import config as C, data as D, modeling as MD, bonus as B\n"
        "for module in (C, D, MD, B):\n"
        "    importlib.reload(module)\n"
        "assert torch.cuda.is_available(), 'A live CUDA GPU is required.'\n"
        "assert torch.cuda.device_count() == 1, 'Use the original core session with one visible GPU.'\n"
        "B.require_weights(C.SFT_MERGED)\n"
        "if RUN_BENCHMARK:\n"
        "    B.require_weights(C.DPO_ADAPTER, adapter=True)\n"
        "mismatch = D.split_mismatch(C.PREF_DIR, C.DPO_ADAPTER)\n"
        "assert mismatch is None, mismatch\n"
        "for folder in (WORK, Path('/tmp')):\n"
        "    free = shutil.disk_usage(folder).free / 1024**3\n"
        "    print(f'{folder}: {free:.2f} GiB free')\n"
        "    assert free >= (0.15 if folder == WORK else 2.0), f'Not enough disk space: {folder}; back up before any cleanup.'\n"
        "print('Ready. Core weights/judge files are not overwritten; no API calls in bonus training.')"
    ))
    cells.extend([
        md("## 1. NB3b — 5 biến thể\n\nNếu một run lỗi, tải kết quả đã có trước; sửa lỗi rồi chạy lại cell. Run hoàn tất được tái dùng khi manifest khớp."),
        code("if RUN_VARIANTS:\n    variants_results = B.run_variants()\nelse:\n    print('NB3b skipped')"),
        code(RELEASE_GPU),
        md("## 2. β-sweep — 0.05 / 0.1 / 0.5\n\nTất cả học lại từ SFT trên cùng 800 cặp; adapter core không thay đổi. Margin có nhân β nên không xếp hạng bằng margin thô."),
        code("if RUN_BETA_SWEEP:\n    beta_results = B.run_beta_sweep()\n    from IPython.display import Image, display\n    display(Image(filename=str(C.SCREENSHOTS / 'bonus-beta-sweep.png')))\nelse:\n    print('Beta sweep skipped')"),
        code(RELEASE_GPU),
        md("## 3. NB6 benchmark — tùy chọn, mặc định bỏ qua\n\nDùng SFT và **DPO core** để đo IFEval/GSM8K/Global-MMLU-vi. T4 batch=1; cần trọng số DPO core. Cài lm-eval chỉ khi bật mục này; chạy trong process riêng."),
        code(
            "if RUN_BENCHMARK:\n"
            "    import subprocess\n"
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'lm-eval[ifeval,math]>=0.4.13,<0.5'])\n"
            "    env = dict(os.environ, BENCH_BATCH='1')\n"
            "    subprocess.check_call([sys.executable, str(WORK / 'notebooks/06_benchmark.py')], cwd=str(WORK), env=env)\n"
            "    from IPython.display import Image, display\n"
            "    display(Image(filename=str(C.SCREENSHOTS / '07-benchmark-comparison.png')))\n"
            "else:\n"
            "    print('NB6 skipped; set RUN_BENCHMARK=True when ready.')"
        ),
        md("## 4. Tải bằng chứng bonus\n\nKhông cập nhật phản tư bằng số liệu giả. Tải ZIP bên dưới và notebook có output, gửi lại để điền §5 (≥100 từ), §8; nếu chạy NB6 thì thêm §7 (≥150 từ). Giữ nguyên bằng chứng core."),
        code(
            "from zipfile import ZipFile, ZIP_DEFLATED\n"
            "evidence = set()\n"
            "for folder, suffixes in (('submission/screenshots', {'.png'}), ('data/pref', {'.parquet'}), ('data/eval', {'.json', '.jsonl'}), ('adapters', {'.json'})):\n"
            "    evidence.update(p for p in (WORK / folder).rglob('*') if p.is_file() and p.suffix in suffixes and p.name != 'tokenizer.json' and 'checkpoints' not in str(p))\n"
            "evidence.add(WORK / 'models/sft-merged/config.json')\n"
            "archive = Path('/kaggle/working/lab22-bonus-evidence.zip')\n"
            "with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:\n"
            "    for path in sorted(evidence):\n"
            "        bundle.write(path, arcname=path.relative_to(WORK).as_posix())\n"
            "print(f'Download {archive} ({len(evidence)} files) and this notebook with outputs.')\n"
            "print('Bonus weights in /tmp are NOT included; back them up separately if needed.')"
        ),
    ])
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the Kaggle bundle is stale")
    args = parser.parse_args()
    stale = []
    for target, notebook in ((TARGET, render()), (REJUDGE_TARGET, render_rejudge()), (BONUS_TARGET, render_bonus())):
        if args.check:
            if not target.exists() or json.loads(target.read_text(encoding="utf-8")) != notebook:
                stale.append(str(target.relative_to(REPO)))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {target.relative_to(REPO)} ({len(notebook['cells'])} cells)")
    if stale:
        print(f"Stale notebooks: {stale}; run python scripts/build_kaggle.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
