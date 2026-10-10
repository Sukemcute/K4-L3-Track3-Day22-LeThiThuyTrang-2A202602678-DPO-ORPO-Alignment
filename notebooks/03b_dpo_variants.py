# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3b — DPO / RPO / DPO-norm / LD-DPO / ORPO (bonus +8)
#
# Cùng SFT reference, cùng seed, cùng train subset (T4: 300 cặp), cùng 100 cặp
# eval và 20 prompt probe. Baseline DPO được train lại trên subset này, không
# lấy kết quả DPO core đã học 800 cặp để so thiếu công bằng.
#
# | Run | Loss | Ý tưởng |
# |---|---|---|
# | DPO | sigmoid | Baseline có reference |
# | RPO | sigmoid + sft, trọng số 1:1 | Thêm NLL chosen |
# | DPO-norm | sigmoid_norm | Chuẩn hóa log-prob theo token; vẫn có reference |
# | LD-DPO | sigmoid, ld_alpha=0.5 | Giảm trọng số phần token dài dư |
# | ORPO | experimental ORPO, beta=0.1 | SFT + odds-ratio, không reference |
#
# ORPO cũng bắt đầu từ SFT để chỉ thay đổi loss. Reward ORPO khác nghĩa/thang
# với DPO; không xếp hạng mô hình bằng margin thô. Accuracy sở thích không
# phải win rate judge. Các cặp probe và output được lưu để đọc chất lượng.
#
# Mỗi run lưu ngay evidence và adapter; nếu run sau lỗi, chạy lại chỉ bỏ qua
# run hoàn tất khi manifest (split, seed, reference, config, probe) vẫn khớp.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))
import unsloth  # noqa: F401
from lab22 import bonus as B

# %% [markdown]
# ## 1. Chạy tuần tự và vẽ bảng tổng hợp
#
# Artifact: adapters/variants/variants_summary.json,
# data/eval/bonus-variants-*.json (history, config và output),
# submission/screenshots/03b-variants.png.

# %%
results = B.run_variants()

# %% [markdown]
# ## 2. Phản tư §8 sau khi có số liệu
#
# 1. Biến thể nào thay đổi mean_output_chars nhiều nhất so với baseline DPO
#    cùng subset? Đọc ít nhất hai output; dài hơn không đồng nghĩa tốt hơn.
# 2. Liên hệ cơ chế từng loss với thiên vị độ dài của NB2; không khẳng định
#    cơ chế đã có tác dụng nếu chỉ số không thể hiện rõ.
# 3. Đọc cả chosen/rejected trong history; RPO có giữ chosen cao hơn không?
# 4. Không so trực tiếp margin ORPO với DPO hoặc DPO-norm.
# 5. Muốn dùng adapter làm NB4, trỏ DPO_ADAPTER_OVERRIDE đến weights_dir
#    được ghi trong metrics, không trỏ đến thư mục evidence chỉ có config.
