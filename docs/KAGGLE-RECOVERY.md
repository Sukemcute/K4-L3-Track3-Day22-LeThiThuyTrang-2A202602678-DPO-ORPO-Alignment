# Chạy tiếp NB4 khi ổ working gần đầy

> Cấu hình hiện tại của bản Kaggle đã chuyển sang **OpenRouter** theo lựa chọn của người học.
> Để chấm lại câu trả lời đã lưu, dùng `kaggle/Lab22_OpenRouter_Rejudge.ipynb`,
> thêm secret `OPENROUTER_API_KEY` và chép các cell vào phiên đang chạy.
> Các bước tải RM dưới đây chỉ ghi lại cách xử lý sự cố của lần chạy local trước đó.

Trường hợp đã gặp: SFT/DPO hoàn tất, NB4 đã lưu `side_by_side.jsonl`,
giám khảo Qwen3 đã được nạp, nhưng tải giám khảo Llama bị lỗi Xet
`Internal Writer Error: Background writer channel closed`.
Kiểm tra cho thấy `/kaggle/working` còn 0,29 GiB, `/tmp` còn hơn 1.000 GiB.

Giữ nguyên phiên hiện tại và thêm cell dưới đây ngay trước cell chấm điểm
(cell bắt đầu bằng `provider = C.JUDGE_PROVIDER`). Cell tải riêng giám khảo
Llama vào `/tmp`, sau đó dùng thư mục model cục bộ cho giám khảo này.
Nó không xóa hoặc di chuyển bất kỳ artifact/cache nào đã có.

```python
from huggingface_hub import snapshot_download

llama_judge = snapshot_download(
    repo_id="Skywork/Skywork-Reward-V2-Llama-3.2-3B",
    cache_dir="/tmp/lab22-judge-cache",
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"],
    max_workers=2,
)
C.JUDGE_RM_MODELS = [C.JUDGE_RM_MODELS[0], llama_judge]
print("Giám khảo Llama đã tải vào:", llama_judge)
```

Chạy lại cell chấm điểm, rồi các cell còn lại để tạo `judge_summary.json`
và `lab22-evidence.zip`. Cell chấm sẽ chấm lại cả hai giám khảo;
không cần chạy lại NB1/NB3 hoặc cell sinh câu trả lời của NB4.
Trong kết quả, tên giám khảo Llama sẽ là đường dẫn snapshot cục bộ;
model vẫn là `Skywork/Skywork-Reward-V2-Llama-3.2-3B`.

Nếu kernel đã restart, cần nạp lại biến cấu hình và câu trả lời đã lưu;
cell trên dành cho phiên hiện tại còn các biến `C`, `J`, `records`, `OUTPUTS_SHA`.
Nếu phiên bị mất cả artifact thì không thể dùng cách này để khôi phục trọng số.

Giám khảo Qwen3 trong lần chạy này chỉ đạt sanity accuracy 50%, dưới ngưỡng
80% của lab. Hãy đọc sanity của giám khảo Llama sau khi chạy tiếp;
không kết luận chất lượng từ một giám khảo không đạt sanity.

Bản notebook mới sinh bằng `python scripts/build_kaggle.py` đặt toàn bộ
cache Hugging Face vào `/tmp/lab22-hf-cache` ngay trước khi import thư viện.
Không chỉ sửa `HF_HOME` ở giữa phiên rồi kỳ vọng thư viện đã import đổi cache.

Tham khảo: [Hugging Face download cache](https://huggingface.co/docs/huggingface_hub/en/guides/download),
[Xet có thể che khuất lỗi hết ổ đĩa](https://github.com/huggingface/xet-core/issues/763).
