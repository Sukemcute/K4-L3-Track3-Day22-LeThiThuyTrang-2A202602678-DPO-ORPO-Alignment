# Bằng chứng Kaggle đã nhập vào repository

- Notebook đã chạy, giữ nguyên output: `kaggle/lab22-dpo-t4-kaggle (2).ipynb`.
- ZIP nguồn lúc nhập: `lab22-evidence (1).zip` (20 file); không cần giữ ZIP trong repo sau khi giải nén/kiểm tra, không cần nộp ZIP.
- Kết quả đã giải nén vào `adapters/`, `models/sft-merged/`, `data/`, `submission/screenshots/`.
- Giám khảo cuối: `openrouter:google/gemini-2.5-flash`.
- SHA-256 của `side_by_side.jsonl`:
  `37ffa9dca872fe26617826d8d88eb1a4e736178af7f288beaced3380e3e4bfe0`.
  Hash trong kết quả API và summary đều khớp; hai file Parquet khớp `adapters/dpo/split.json`.
- 800 cặp train / 100 eval không trùng prompt; 50 prompt held-out NB4 đều thuộc eval.
- Giữ kết quả local ở `judge_results_rm.json` để đối chiếu; summary cuối không dùng RM làm giám khảo chính.

## Kiểm tra trên máy mới

```bash
python scripts/verify.py
python -m pytest -q scripts/
python scripts/build_kaggle.py --check
```

`verify.py` nhận diện chính xác đường dẫn reference của notebook Kaggle/Colab, đồng thời yêu cầu
metadata reference, merged config và fingerprint split. Không chấp nhận mọi đường dẫn chỉ vì
có cùng đuôi `models/sft-merged`; không sửa adapter config gốc. Đây là kiểm tra bằng chứng,
không phải xác minh nội dung trọng số.

## Chạy lại GPU

Import `kaggle/Lab22_DPO_T4_Kaggle.ipynb`, chọn T4, bật Internet và cấp quyền secret
`OPENROUTER_API_KEY`. Notebook mẫu hiện dùng OpenRouter ngay từ đầu, cache ở `/tmp`.
Notebook có output là lịch sử đã chạy RM trước, sau đó chấm lại API; không thay thế nó bằng
notebook mẫu không có output.

ZIP cố ý không có trọng số. Muốn tiếp tục inference/training trên checkpoint đã fine-tune,
cần sao lưu riêng trọng số từ Kaggle. Không commit các file trọng số lớn hoặc API key.
Config nhỏ `models/sft-merged/config.json` được phép theo dõi bằng Git để bộ kiểm tra chạy
được sau khi clone repository.
`.gitattributes` giữ nguyên byte JSONL và Parquet, tránh chuyển đổi xuống dòng trên Windows
làm lệch hash đã lưu trong kết quả chấm.

Trước khi nộp, người học cần đọc lại phản tư, xác nhận tên có dấu và thông tin khóa/mã,
commit/push các bằng chứng và nộp URL repository public lên LMS. Chưa thực hiện push hoặc nộp LMS.
