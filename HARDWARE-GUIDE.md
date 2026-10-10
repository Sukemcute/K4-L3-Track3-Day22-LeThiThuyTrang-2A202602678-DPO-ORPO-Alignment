# Hướng dẫn phần cứng — chọn tier phù hợp

## 1. VRAM cho DPO với LoRA

Khi mô hình đang học dùng PEFT/LoRA, TRL **không** nạp bản sao thứ hai của mô hình làm tham chiếu. Lab này
còn đi thêm một bước: **tính sẵn** log-xác suất của mô hình tham chiếu (`precompute_ref_log_probs=True`) bằng
mô hình SFT đã gộp trước khi huấn luyện, nên lúc huấn luyện GPU chỉ giữ một mô hình 4-bit cộng trọng số LoRA.

DPO vẫn tốn bộ nhớ hơn SFT vì mỗi bước phải đưa cả câu `chosen` **và** câu `rejected` qua mô hình: bộ nhớ
activation xấp xỉ gấp đôi SFT với cùng `max_length` và batch. Không phải "gấp đôi trọng số".

Số liệu ước lượng (mô hình gốc 4-bit, LoRA r=16, bật gradient checkpointing; chưa đo trên đúng cấu hình này):

| Mô hình gốc | Trọng số (4-bit) | Đỉnh bộ nhớ thường gặp khi chạy DPO | Chạy được trên |
|---|---:|---:|---|
| Qwen3-4B-Instruct-2507 | ~3 GB | ~9–12 GB với max_len 768, batch 1 | Colab T4 16 GB, RTX 3060 12 GB |
| Gemma 4 E4B (phương án thay thế) | ~4–5 GB | ~11–14 GB | T4 với `MAX_LEN=512` |
| Qwen3-8B | ~6 GB | ~16–20 GB với max_len 1024, batch 2 | L4 24 GB, A100, RTX 3090/4090 |

Hai thời điểm bộ nhớ tăng vọt cần lưu ý:

- **Bước gộp ở NB1** và **bước xuất GGUF ở NB5** nạp mô hình ở dạng 16-bit (4B ≈ 8 GB): vẫn ổn trên T4.
- **NB7 GRPO** sinh G câu trả lời cho mỗi câu hỏi; giảm `G` hoặc `max_completion_length` nếu hết bộ nhớ.

## 2. Chọn tier

| Tài nguyên bạn có | Tier | Cách chạy |
|---|---|---|
| Colab T4 miễn phí | **T4** | `colab/Lab22_DPO_T4.ipynb` |
| Kaggle T4×2 | T4 (dùng một GPU) | `kaggle/Lab22_DPO_T4_Kaggle.ipynb` (Run All: NB0–NB4) |
| Colab Pro L4 / A100 | **BigGPU** | `colab/Lab22_DPO_BigGPU.ipynb` |
| GPU laptop 12–23 GB | T4 | `setup-laptop.sh` + `make pipeline` |
| GPU ≥ 24 GB | BigGPU | `COMPUTE_TIER=BIGGPU make pipeline` |
| Không có GPU | — | NB0 chạy trên CPU; các phần còn lại cần GPU (dùng Colab) |

Nếu hết bộ nhớ GPU (OOM): giảm `MAX_LEN` (768 → 512), sau đó tăng `gradient_accumulation_steps` trong
`lab22/config.py`, cuối cùng mới hạ tier.

## 3. Ổ đĩa

Trọng số gốc 3–6 GB, mô hình SFT đã gộp 8–16 GB (16-bit), cache Hugging Face ~10 GB, GGUF Q4_K_M 2,5–5 GB.
Nếu chạy trên máy riêng, nên để trống **40 GB**. Colab cấp khoảng 100 GB.

## 4. Mạng

Cần truy cập Hugging Face để tải mô hình và dữ liệu. Giám khảo mặc định của NB4 là hội đồng hai reward model chạy
local (tải từ Hugging Face, ~8 GB + ~6,5 GB, nạp lần lượt từng cái). Giám khảo API (tuỳ chọn) cần kết nối HTTPS
tới nhà cung cấp đã chọn. Nếu đã chọn giám khảo API nhưng thiếu khoá, NB4 dừng để cấu hình.
Bản Kaggle dùng OpenRouter qua Kaggle Secrets; notebook chấm lại chỉ cần Internet và câu trả lời đã lưu, không cần GPU.

## 5. Máy Apple Silicon

bitsandbytes 4-bit và các kernel CUDA của Unsloth không được hỗ trợ trên MPS, nên các notebook dùng GPU chỉ nhắm
tới CUDA. NB0 và `make test` chạy được trên Mac. Nếu muốn làm dự án mở rộng chỉ dùng máy Apple, MLX-LM có công
cụ LoRA riêng; phần đó nằm ngoài phạm vi lab này.
