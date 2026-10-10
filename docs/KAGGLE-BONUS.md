# Chạy bonus từ phiên Kaggle còn trọng số

Không Restart Session, không Run All lại notebook core. ZIP evidence không chứa trọng số;
một notebook Kaggle mới không tự truy cập thư mục của notebook cũ.

## Lộ trình

| Phần | Điểm rubric | Trạng thái |
|---|---:|---|
| Chấm chéo RM và Gemini | +4 | Đã có bằng chứng: agreement 44/58 = 75,86% |
| NB3b: DPO/RPO/DPO-norm/LD-DPO/ORPO | +8 | Đã chuẩn bị mã; chưa có kết quả GPU |
| β-sweep: 0,05 / 0,1 / 0,5 | +6 | Đã chuẩn bị mã; chưa có kết quả GPU |
| NB6: IFEval/GSM8K/Global-MMLU-vi | +6 | Tùy chọn, tắt mặc định |

Ba mục đầu có tổng 18 điểm tiềm năng. Nếu hoàn thành thêm NB6, tổng thô là 24 nhưng rubric
chỉ tính tối đa 20 bonus. Điểm thực tế phụ thuộc bằng chứng và phần giải thích, không chỉ chạy code.
Chưa làm GGUF, GRPO hoặc upload HF Hub trong quy trình này.

## Cách chạy

1. Giữ nguyên notebook Kaggle đã chạy core và còn trọng số.
2. Mở `kaggle/Lab22_Bonus_T4_Kaggle.ipynb`, **copy các cell vào cuối notebook cũ**.
   Nếu mở file trong một notebook Kaggle mới để copy, không chạy ở phiên mới trống dữ liệu.
3. Chỉ chạy các cell bonus vừa thêm, từ cell cấu hình xuống dưới:
   `RUN_VARIANTS=True`, `RUN_BETA_SWEEP=True`, `RUN_BENCHMARK=False`.
4. Preflight kiểm tra CUDA, các shard SFT thật, fingerprint train/eval và dung lượng ổ đĩa.
   Không thay reference SFT bằng mô hình gốc nếu thiếu trọng số. Không tự xóa cache/checkpoint.
5. NB3b chạy năm lượt trên cùng 300 cặp đầu của split train (tier T4), cùng 100 eval,
   cùng 20 prompt probe, seed, lr, epoch, LoRA. Baseline DPO cũng học lại 300 cặp.
6. β-sweep chạy ba lượt mới trên **toàn bộ 800 train / 100 eval**. Không ghi đè DPO core.
7. Chạy cell cuối để tải `lab22-bonus-evidence.zip` và tải notebook đã giữ output.
   Gửi lại để kiểm tra số liệu và cập nhật phản tư §5 (ít nhất 100 từ), §8.

Đây là tám lượt fine-tune mới, không phải chỉ tám lần forward. Thời gian phụ thuộc GPU,
độ dài dữ liệu và phiên Kaggle; không cam kết một thời lượng cố định. Không cần OpenRouter
key cho NB3b hoặc sweep, không phát sinh các lượt gọi API trong hai phần này.

## Khi lỗi hoặc cần chạy tiếp

Mỗi lượt thành công lưu ngay `data/eval/bonus-variants-*.json` hoặc
`data/eval/bonus-beta-*.json`, gồm manifest, history train/eval, các output probe và metrics.
Chạy lại cell của phần lỗi sẽ tái dùng lượt hoàn tất **chỉ khi** dữ liệu, reference, cấu hình,
phiên bản thư viện và probes khớp manifest. Reference được nhận diện bằng config/tokenizer hash
và tên/kích thước/mtime của shard, không phải checksum đầy đủ của toàn bộ trọng số.
Nếu manifest khác, chương trình dừng để tránh trộn thí nghiệm; giữ bản cũ và chọn nơi lưu mới
trong helper trước khi bắt đầu thí nghiệm khác.

Nếu một phần lỗi, vẫn có thể chạy riêng cell xuất ZIP cuối cùng để tải bằng chứng các lượt đã xong.
Lượt đang train dở không có checkpoint optimizer định kỳ; nó phải chạy lại từ SFT, không tự
resume từ giữa step. Bảng/biểu đồ hoàn chỉnh chỉ được tạo sau khi đủ các lượt tương ứng.

Adapter mới lưu tại `/tmp/lab22-bonus-weights/<stage>/<run>`; working chỉ giữ JSON/ảnh nhỏ.
Thư mục này mất khi phiên kết thúc và không có trong ZIP evidence. Sao lưu riêng nếu cần
dùng các adapter để suy luận hoặc chấm NB4. Không commit trọng số lớn lên GitHub.
Đường `weights_dir` trong metrics là vị trí thực tế của adapter; thư mục config trong repo
không đủ để load adapter.

## NB6 nếu muốn làm thêm

Sau khi NB3b và sweep xong, đổi `RUN_VARIANTS=False`, `RUN_BETA_SWEEP=False`,
`RUN_BENCHMARK=True`; chạy lại cell cấu hình rồi cell NB6. Không cần chạy lại các cell
`%%writefile` nếu vẫn trong phiên vừa chạy bonus. NB6 dùng SFT và **DPO core**, không lấy
biến thể có điểm cao nhất trên eval để tuyên bố cải thiện benchmark.

Cell NB6 mới cài lm-eval khi bật; chạy đánh giá trong process riêng với batch 1 trên T4,
cache tải về trong `/tmp`. Đây vẫn là một workload GPU bổ sung, không bảo đảm tránh mọi OOM.
Giữ đúng `--apply_chat_template`, GSM8K dùng `--fewshot_as_multiturn`. Limit Global-MMLU-vi
tính theo môn, không phải tổng toàn bộ môn.

Kết quả: `data/eval/benchmark_results.json`, `submission/screenshots/07-benchmark-comparison.png`.
Nếu lỗi ở benchmark sau, `benchmark_progress.json` giữ các bộ đã hoàn thành; code hiện không
tự bỏ qua từng benchmark đã chạy khi chạy lại. Xuất lại ZIP, giữ output, rồi viết §7 ít nhất
150 từ, thảo luận chênh lệch với stderr; chưa có số liệu thì không điền điểm giả.

## Đọc kết quả đúng

- So độ dài các biến thể với **DPO baseline cùng subset**, rồi đọc output thật.
- Reward accuracy của ORPO và DPO có cách định nghĩa reward khác nhau; không xem đây là
  đánh giá chất lượng độc lập. Margin thô giữa các loss không so trực tiếp được.
- Với sweep, margin đã nhân β. β lớn có thể tạo margin số lớn mà chất lượng không tốt hơn.
- `training_seconds` đo `trainer.train()`; `peak_allocated_gib_train_eval` đo CUDA allocated
  trong train/eval sau khi nạp mô hình, không phải toàn bộ VRAM reserved hoặc peak lúc load.
- Chỉ đánh dấu bonus hoàn tất sau khi đủ run, ảnh, JSON, notebook có output và phần giải thích.

API loss được đối chiếu với [DPOTrainer TRL 1.13](https://huggingface.co/docs/trl/v1.13.0/dpo_trainer)
và [ORPOTrainer TRL 1.13](https://github.com/huggingface/trl/blob/v1.13.0/trl/experimental/orpo/orpo_trainer.py).
Môi trường local chỉ kiểm thử CPU/mock; các lượt GPU mới phải được chạy trong phiên Kaggle.
