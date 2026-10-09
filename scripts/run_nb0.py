#!/usr/bin/env python3
"""Execute the CPU-only NB0 cells and save a notebook with real outputs.

    python scripts/run_nb0.py

Uses torch and the standard library, without needing Jupyter or a GPU.
NB0 contains plain Python and text outputs only. For other stages use
the Jupytext/Jupyter pipeline in the Makefile.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

from build_colab import percent_cells

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "notebooks" / "00_dpo_loss_from_scratch.py"


def main() -> None:
    cells = percent_cells(SOURCE)
    namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
    previous_cwd = Path.cwd()
    count = 0
    try:
        os.chdir(REPO)
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            count += 1
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile("".join(cell["source"]), str(SOURCE), "exec"), namespace)
            cell["execution_count"] = count
            for name, buffer in (("stdout", stdout), ("stderr", stderr)):
                value = buffer.getvalue()
                if value:
                    cell["outputs"].append({"output_type": "stream", "name": name, "text": value})
                    print(value, end="")
    finally:
        os.chdir(previous_cwd)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    target = SOURCE.with_suffix(".ipynb")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Saved {target.relative_to(REPO)} ({count} code cells executed)")


if __name__ == "__main__":
    main()
