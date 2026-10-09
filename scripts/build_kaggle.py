#!/usr/bin/env python3
"""Generate the Kaggle T4 notebook for the core lab (NB0–NB4).

    python scripts/build_kaggle.py
    python scripts/build_kaggle.py --check
"""
from __future__ import annotations

import argparse
import json

from build_colab import RELEASE_GPU, REPO, STAGES, code, md, percent_cells, requirements

TARGET = REPO / "kaggle" / "Lab22_DPO_T4_Kaggle.ipynb"
WORKDIR = "/kaggle/working/lab22"


def render() -> dict:
    # GGUF and lm-eval dependencies are used only by bonus notebooks.
    specs = [spec for spec in requirements() if not spec.startswith(("llama-cpp-python", "lm-eval"))]
    cells = [
        md(
            "# Lab 22 — DPO Alignment · Kaggle T4\n\n"
            "Sinh từ `notebooks/*.py` và `lab22/*.py` bằng `scripts/build_kaggle.py`.\n\n"
            "1. Trong Settings, chọn **GPU T4 ×2** và bật **Internet**.\n"
            "2. Chạy notebook trong một phiên mới. Cấu hình dùng một GPU, tier T4.\n"
            "3. **Run All** chạy Setup → NB0 → NB1 → NB2 → NB3 → NB4; không chạy bonus.\n"
            "4. Cuối notebook tạo `/kaggle/working/lab22-evidence.zip`. Tải ZIP và notebook "
            "có output về máy. ZIP chứa bằng chứng nhỏ, không chứa trọng số để tiếp tục training.\n\n"
            "Kết quả được ghi vào `/kaggle/working/lab22`. Mỗi giai đoạn nạp lại artifact "
            "cần thiết từ thư mục này; nếu phiên mất dữ liệu, cần chạy lại từ NB1."
        ),
        md("## A. Setup"),
        code(
            "import os\n"
            "import sys\n"
            "if 'torch' in sys.modules or 'unsloth' in sys.modules:\n"
            "    raise RuntimeError('Restart Session, then Run All from the first cell.')\n"
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
            'os.environ["COMPUTE_TIER"] = "T4"\n'
            'os.environ["HF_HOME"] = "/kaggle/working/hf_cache"\n'
            "# Local reward-model panel: no API key needed.\n"
            'os.environ["JUDGE_PROVIDER"] = "rm"\n'
            "# Optional OpenRouter judge: uncomment all three lines below.\n"
            '# from kaggle_secrets import UserSecretsClient\n'
            '# os.environ["OPENROUTER_API_KEY"] = UserSecretsClient().get_secret("OPENROUTER_API_KEY")\n'
            '# os.environ["JUDGE_PROVIDER"] = "openrouter"; os.environ["JUDGE_MODEL"] = "google/gemini-2.5-flash"\n'
            "# Store API keys in Kaggle Secrets, never in notebook source."
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the Kaggle bundle is stale")
    args = parser.parse_args()
    notebook = render()
    if args.check:
        if not TARGET.exists() or json.loads(TARGET.read_text(encoding="utf-8")) != notebook:
            print("Kaggle notebook is stale; run python scripts/build_kaggle.py")
            return 1
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {TARGET.relative_to(REPO)} ({len(notebook['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
