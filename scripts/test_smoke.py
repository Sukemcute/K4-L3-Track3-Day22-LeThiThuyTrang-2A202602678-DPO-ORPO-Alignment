"""CPU-only structural checks: sources parse, trainer APIs match TRL 1.13 /
transformers 5, and the Colab bundles are in sync with the sources.

Run:  pytest -q scripts/   (or `make test`).
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "00_dpo_loss_from_scratch", "01_sft_mini", "02_preference_data", "03_dpo_train",
    "03b_dpo_variants", "04_compare_and_eval", "05_merge_deploy_gguf", "06_benchmark",
    "07_grpo_bonus",
]
SOURCES = [REPO / "notebooks" / f"{nb}.py" for nb in NOTEBOOKS] + sorted((REPO / "scripts").glob("*.py")) + sorted(
    (REPO / "lab22").glob("*.py")
)


def test_sources_exist_and_parse():
    for p in SOURCES:
        assert p.exists(), f"missing {p}"
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


def test_no_removed_trainer_arguments():
    # tokenizer= (TRL >= 0.13), warmup_ratio (transformers 5), max_prompt_length (TRL 1.x DPOConfig).
    banned = re.compile(r"\btokenizer\s*=\s*tokenizer\b|\bwarmup_ratio\s*=|\bmax_prompt_length\s*=")
    offenders = [str(p.relative_to(REPO)) for p in SOURCES if banned.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"removed trainer arguments in {offenders}"


def test_no_hardcoded_judge_model():
    for p in SOURCES:
        if p.name == "test_smoke.py":
            continue
        text = p.read_text(encoding="utf-8")
        assert "gpt-4o-mini" not in text and "claude-haiku" not in text, f"hard-coded judge id in {p}"


def test_colab_bundles_are_valid_and_current():
    from build_colab import render

    for tier, path in (("T4", "Lab22_DPO_T4.ipynb"), ("BIGGPU", "Lab22_DPO_BigGPU.ipynb")):
        on_disk = json.loads((REPO / "colab" / path).read_text(encoding="utf-8"))
        assert on_disk == render(tier), f"colab/{path} is stale: run `make colab`"


def test_kaggle_bundle_is_current_and_has_only_core_stages():
    from build_kaggle import TARGET, render

    notebook = json.loads(TARGET.read_text(encoding="utf-8"))
    assert notebook == render(), "Kaggle bundle is stale: run python scripts/build_kaggle.py"
    stages = [
        "".join(cell["source"]).split("`", 2)[1]
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("---\n# ⏵")
    ]
    assert stages == [f"notebooks/{stem}.py" for stem in NOTEBOOKS[:4] + ["04_compare_and_eval"]]


def test_openrouter_rejudge_bundle_is_current_and_does_not_load_policy():
    from build_kaggle import REJUDGE_TARGET, render_rejudge

    notebook = json.loads(REJUDGE_TARGET.read_text(encoding="utf-8"))
    assert notebook == render_rejudge()
    # Embedded helper source defines a lazy RM scorer; writing it does not import torch.
    all_code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"]
        if c["cell_type"] == "code" and not "".join(c["source"]).startswith("%%writefile ")
    )
    assert "side_by_side.jsonl" in all_code
    assert 'os.environ["JUDGE_PROVIDER"] = "openrouter"' in all_code
    assert "importlib.reload(C)" in all_code
    assert "MD.load_model" not in all_code and "MD.generate" not in all_code
    assert "import unsloth" not in all_code and "import torch" not in all_code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if source.startswith("%%writefile "):
                source = source.split("\n", 1)[1]
            ast.parse(source)


def test_kaggle_bundle_uses_kaggle_paths_and_valid_python():
    from build_kaggle import WORKDIR, render

    notebook = render()
    all_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "/content/" not in all_source and "google.colab" not in all_source
    assert "colab" not in notebook["metadata"]
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    first = "".join(code_cells[0]["source"])
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in first
    assert 'os.environ["COMPUTE_TIER"] = "T4"' in first
    assert 'os.environ["HF_HOME"] = "/tmp/lab22-hf-cache"' in first
    for variable, folder in (("HF_HUB_CACHE", "hub"), ("HF_DATASETS_CACHE", "datasets"), ("HF_XET_CACHE", "xet")):
        assert f'os.environ["{variable}"] = "/tmp/lab22-hf-cache/{folder}"' in first
    assert "/kaggle/working/hf_cache" not in first
    assert "import torch" not in first
    assert "kaggle_secrets" in first
    installer = "".join(code_cells[1]["source"])
    assert "llama-cpp-python" not in installer and "lm-eval" not in installer
    written = set()
    for cell in code_cells:
        source = "".join(cell["source"])
        if source.startswith("%%writefile "):
            header, source = source.split("\n", 1)
            path = header.removeprefix("%%writefile ")
            assert path.startswith(f"{WORKDIR}/lab22/")
            written.add(Path(path).name)
        ast.parse(source)
    assert written == {p.name for p in (REPO / "lab22").glob("*.py")}
    assert "lab22-evidence.zip" in "".join(code_cells[-1]["source"])
