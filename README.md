# Ngày 22 — Lab căn chỉnh mô hình bằng DPO/ORPO (Track 3)

Lab cho học phần **AICB-P2T3 · Ngày 22 · DPO/ORPO Alignment — từ SFT đến học theo sở thích**.

> Bản K4 cập nhật tháng 10/2026 (xem [`CHANGELOG.md`](CHANGELOG.md)). Mọi thời gian trong tài liệu này là
> ước tính trên Colab T4 miễn phí; máy của bạn có thể nhanh hoặc chậm hơn.

---

## 0. Lab này làm gì?

Một mô hình ngôn ngữ vừa huấn luyện xong thường trả lời "đúng ngữ pháp" nhưng chưa chắc đã trả lời
**theo cách con người thích**. Lab này đi qua hai bước để dạy mô hình điều đó, rồi đo xem nó có tiến bộ thật không:

1. **Dạy mô hình làm theo chỉ dẫn** bằng các cặp "câu hỏi → câu trả lời mẫu" tiếng Việt.
2. **Dạy mô hình chọn câu trả lời tốt hơn** bằng các cặp "câu trả lời được chọn / câu trả lời bị loại".
3. **Chấm điểm**: đưa cùng một câu hỏi cho bản trước và bản sau bước 2, nhờ "giám khảo tự động" chọn câu hay hơn.

### Các thuật ngữ sẽ gặp

| Thuật ngữ | Nghĩa trong lab này |
|---|---|
| **SFT** (Supervised Fine-Tuning) | Huấn luyện có giám sát: cho mô hình xem câu hỏi kèm câu trả lời mẫu để nó bắt chước. |
| **Dữ liệu sở thích** (preference) | Mỗi mẫu gồm một câu hỏi và hai câu trả lời: `chosen` (được chọn, tốt hơn) và `rejected` (bị loại). |
| **DPO** (Direct Preference Optimization) | Thuật toán dạy mô hình tăng xác suất câu `chosen` và giảm xác suất câu `rejected`, không cần huấn luyện mô hình chấm điểm riêng. |
| **Mô hình tham chiếu** (reference) | Bản SFT được giữ cố định. DPO so mô hình mới với bản này để mô hình không "đi quá xa" khỏi điểm xuất phát. |
| **β (beta)** | Hệ số điều chỉnh mức được phép đi xa khỏi mô hình tham chiếu. β nhỏ thì mô hình thay đổi mạnh hơn. |
| **LoRA / adapter** | Thay vì sửa toàn bộ mô hình, ta chỉ huấn luyện một phần nhỏ gắn thêm (adapter). Nhẹ, vừa GPU T4. |
| **Reward ngầm** (`rewards/chosen`, `rewards/rejected`) | Điểm DPO tự tính cho mỗi câu trả lời: mô hình mới thích câu đó hơn mô hình tham chiếu bao nhiêu. |
| **Margin** (`rewards/margins`) | Hiệu số reward của câu `chosen` trừ câu `rejected`. Margin tăng nghĩa là mô hình phân biệt tốt hơn. |
| **Held-out** | Phần dữ liệu để riêng, **không** dùng khi huấn luyện, chỉ dùng để kiểm tra. Lab chia theo câu hỏi nên không có câu hỏi nào nằm ở cả hai phía. |
| **Giám khảo** (judge) | Chương trình chấm xem câu trả lời của SFT hay SFT+DPO tốt hơn. Mặc định là hội đồng hai mô hình chấm điểm chạy ngay trên Colab, không cần khoá API. |
| **Win rate** | Tỉ lệ câu hỏi mà bản DPO thắng bản SFT. 0.5 nghĩa là ngang nhau. |
| **Khoảng tin cậy 95%** (CI) | Khoảng giá trị mà win rate thật có thể nằm trong. Nếu khoảng này chứa 0.5 thì **chưa đủ bằng chứng** DPO tốt hơn. |

Mô hình dùng trên T4: **Qwen3-4B** (nén 4-bit). Dữ liệu: 1.000 mẫu SFT tiếng Việt, 800 cặp sở thích tiếng
Việt để huấn luyện và 100 cặp để kiểm tra.

---

## 1. Chuẩn bị (Colab, không cần cài gì)

1. Tải file [`colab/Lab22_DPO_T4.ipynb`](colab/Lab22_DPO_T4.ipynb) về máy, rồi mở [Google Colab](https://colab.research.google.com)
   → **Tệp → Tải sổ tay lên** → chọn file vừa tải.
2. Chọn GPU: **Thời gian chạy → Thay đổi loại thời gian chạy → T4 GPU → Lưu**.
3. Chạy các cell đầu tiên (phần cài đặt). Chúng đặt cấu hình (cell đầu tiên, gọi là **cell cài đặt**), cài thư
   viện, tạo thư mục làm việc `/content/lab22` và ghi các file mã nguồn của lab vào đó. Bạn không cần tải repo về.
4. Chạy lần lượt các cell **từ trên xuống**, không bỏ cell nào. Phần bắt buộc kết thúc ở **NB4**; các phần sau là bonus.
   Nếu gặp lỗi không có GPU, quay lại bước 2.

> **Quan trọng — Colab xoá mọi file khi hết phiên.** Trước khi đóng tab hoặc hết giờ GPU, mở bảng **Tệp**
> (biểu tượng thư mục bên trái), vào `/content/lab22` và tải về máy:
> - thư mục `submission/screenshots/` (các ảnh biểu đồ),
> - thư mục `data/eval/` (kết quả chấm),
> - các file `.json` trong `adapters/dpo/` (số liệu huấn luyện; **không** cần tải file trọng số `.safetensors`).
>
> Nếu mất phiên giữa chừng, bạn phải chạy lại từ NB1.

Muốn chạy trên laptop/máy chủ có GPU ≥ 12 GB, hoặc dùng A100/L4: xem [`docs/reference.md`](docs/reference.md).

### Chạy trên Kaggle

Import [`kaggle/Lab22_DPO_T4_Kaggle.ipynb`](kaggle/Lab22_DPO_T4_Kaggle.ipynb) vào Kaggle,
chọn **GPU T4 ×2**, bật **Internet**, rồi **Run All** trong một phiên mới.
Bản này dùng một GPU, tier T4, chạy phần bắt buộc NB0 → NB4 và bỏ các phần bonus.
Đường dẫn và các cell `%%writefile` đã dùng `/kaggle/working/lab22`; không cần tự sửa.
Giám khảo mặc định chạy local, không cần API key. Nếu dùng API, cấu hình bằng Kaggle Secrets
theo hướng dẫn trong cell đầu tiên.

Cuối notebook tạo `/kaggle/working/lab22-evidence.zip` chứa ảnh, dữ liệu và JSON kết quả.
Tải ZIP cùng notebook có output về máy để hoàn thiện `submission/REFLECTION.md` và nộp bài.
ZIP không chứa trọng số; muốn tiếp tục training ở phiên khác cần sao lưu riêng model/adapter.
Sinh lại notebook sau khi sửa mã nguồn: `python scripts/build_kaggle.py`;
kiểm tra đồng bộ: `python scripts/build_kaggle.py --check`.

---

## 2. Từng bước một

Tổng thời gian phần bắt buộc khoảng **1,5–2 giờ** trên T4. Mỗi notebook (NB) dưới đây tương ứng một phần
trong file Colab, theo đúng thứ tự.

### NB0 — Tự viết hàm loss của DPO (~10 phút, chạy được trên CPU)

**Mục đích:** hiểu công thức DPO trước khi dùng thư viện.

**Việc cần làm:** điền hàm `my_dpo_loss` (chỗ có `# TODO`). Hàm nhận log-xác suất của câu `chosen` và
`rejected` dưới mô hình đang học (`pc`, `pr`) và dưới mô hình tham chiếu (`rc`, `rr`), rồi trả về loss trung bình.
Gợi ý: dùng `torch.nn.functional.logsigmoid`; công thức có ngay trong notebook.

**Xong khi:** các `assert` chạy qua. Một kiểm tra hay: khi mô hình mới giống hệt mô hình tham chiếu, loss phải
bằng `log 2 ≈ 0.693`.

**Câu hỏi cần trả lời:** vì sao margin có thể tăng trong khi xác suất của câu `chosen` lại giảm? (gợi ý: nếu
câu `rejected` giảm nhanh hơn thì sao?)

### NB1 — SFT: dạy mô hình làm theo chỉ dẫn (~15–25 phút)

**Mục đích:** tạo điểm xuất phát cho DPO.

**Việc cần làm:** chạy các cell. Notebook lấy 1.000 mẫu hỏi–đáp tiếng Việt, huấn luyện một adapter LoRA, rồi
gộp adapter vào mô hình gốc để được `models/sft-merged/`. Bản gộp này chính là **mô hình tham chiếu** cho NB3.

**Cần thấy:** đường loss trong ảnh `02-sft-loss.png` đi xuống. Nếu loss không giảm, kiểm tra lại GPU và dữ liệu
trước khi sang bước sau.

**Kết quả:** `models/sft-merged/`, ảnh `submission/screenshots/02-sft-loss.png`.

### NB2 — Chuẩn bị dữ liệu sở thích (~2 phút)

**Mục đích:** có dữ liệu sạch và biết trước điểm yếu của nó.

**Việc cần làm:**
1. Chạy cell tải dữ liệu. Notebook lọc các cặp tiếng Việt, chia **800 cặp huấn luyện** và **100 cặp held-out**
   theo câu hỏi, và tự kiểm tra hai phần không trùng câu hỏi nào.
2. Đọc kỹ **3 cặp mẫu** được in ra. Tự hỏi: câu `chosen` có thật sự tốt hơn không, hay chỉ dài hơn?
3. Xem phần "Thiên vị độ dài": notebook đo tỉ lệ cặp có câu `chosen` dài hơn câu `rejected`. Nếu tỉ lệ này
   cao, DPO dễ học cách "viết dài cho được điểm" thay vì viết hay. Ghi lại con số này, NB4 sẽ cần.

**Kết quả:** ảnh `02b-pref-length.png`, dữ liệu trong `data/pref/`.

### NB3 — Huấn luyện DPO (~40–60 phút, bước lâu nhất)

**Mục đích:** dạy mô hình SFT ưu tiên câu `chosen`.

**Việc cần làm:** chạy các cell rồi chờ. Trước khi huấn luyện, notebook tính sẵn log-xác suất của mô hình
tham chiếu (vài phút đầu không có thanh tiến trình huấn luyện, đó là bình thường). Sau đó chạy khoảng 100 bước,
cứ 25 bước lại đánh giá trên tập held-out một lần.

**Cần thấy:** ảnh `03-dpo-reward-curves.png` vẽ riêng `chosen` và `rejected`, cho cả tập huấn luyện và
held-out. Cuối notebook có một dòng **chẩn đoán tự động**. Cách đọc biểu đồ:

| Bạn thấy | Tên gọi | Ý nghĩa |
|---|---|---|
| `chosen` tăng, `rejected` giảm, margin tăng | **Đúng kỳ vọng** (INTENDED) | DPO hoạt động như lý thuyết. |
| Margin tăng nhưng `chosen` cũng giảm (`rejected` giảm nhanh hơn) | **Dịch chuyển xác suất** (LIKELIHOOD DISPLACEMENT) | Rất hay gặp với DPO. Không hẳn là hỏng, nhưng bạn phải giải thích. |
| Margin ≤ 0 trên held-out | **Thất bại** (FAILURE) | Mô hình không học được, hoặc chỉ học thuộc tập huấn luyện. |
| Không rõ ràng | **Chưa kết luận** (AMBIGUOUS) | Mô tả những gì bạn thấy và nêu giả thuyết. |

Reward bắt đầu từ 0 vì lúc đầu mô hình mới trùng với mô hình tham chiếu. Chú ý so **đường held-out với đường
huấn luyện**: nếu chỉ đường huấn luyện tăng còn held-out đứng yên, mô hình đang học thuộc.

**Kết quả:** `adapters/dpo/` (có `dpo_metrics.json`), ảnh `03-dpo-reward-curves.png`.

### NB4 — So sánh SFT và SFT+DPO, chấm tự động (~20–30 phút)

**Mục đích:** biết DPO có làm câu trả lời tốt hơn thật không.

**Việc cần làm:** chạy các cell. Notebook cho cả hai bản trả lời **8 câu hỏi cố định** (4 câu về độ hữu ích,
4 câu về an toàn) và **ít nhất 50 câu held-out**, rồi nhờ hội đồng hai mô hình chấm điểm chọn câu hay hơn.
Không cần khoá API.

**Cần thấy:** ảnh `04-side-by-side-table.png` (bảng so sánh 8 câu) và file `data/eval/judge_summary.json`.
Cách đọc file kết quả:

- **Khoảng tin cậy chứa 0.5** ⇒ chưa đủ bằng chứng DPO tốt hơn SFT. Đó vẫn là một kết quả hợp lệ, hãy viết thật.
- **`sanity_accuracy` dưới 0.8** ⇒ giám khảo đọc tiếng Việt chưa tốt, đừng tin win rate.
- **`per_judge`**: win rate của từng giám khảo. Cả hai giám khảo đều thuộc họ Skywork, cùng họ với mô hình đã
  gán nhãn dữ liệu huấn luyện, nên có thể thiên vị DPO (hiện tượng "rò rỉ sở thích"). Nếu một giám khảo cho DPO
  thắng cao hơn hẳn giám khảo kia, hãy nêu điều đó.
- **`longer_answer_won_frac` gần 1** và câu DPO dài hơn câu SFT ⇒ có thể DPO chỉ học viết dài. Xem thêm
  `length_matched_win_rate` (chỉ so các cặp dài gần bằng nhau).

**Kết quả:** `data/eval/side_by_side.jsonl`, `data/eval/judge_summary.json`, ảnh `04-side-by-side-table.png`.

### Phần bonus (không bắt buộc, tối đa +20 điểm)

| Phần | Nội dung | Điểm |
|---|---|---:|
| NB3b | So 5 biến thể: DPO, RPO, DPO-norm, LD-DPO, ORPO; biến thể nào đổi độ dài câu trả lời nhiều nhất và vì sao | +8 |
| NB5 | Xuất mô hình sang định dạng GGUF 4-bit và chạy thử bằng llama.cpp | +4 |
| NB6 | Chạy bộ đo chuẩn IFEval / GSM8K / Global-MMLU tiếng Việt | +6 |
| NB7 | GRPO: học tăng cường với phần thưởng kiểm chứng được trên bài toán tiếng Việt | +8 |
| β-sweep | Huấn luyện lại với β = 0,05 / 0,1 / 0,5 và so sánh | +6 |
| Chấm chéo | Chấm thêm bằng giám khảo API khác họ, báo mức đồng thuận | +4 |
| HF Hub | Đẩy adapter và thẻ mô tả mô hình lên Hugging Face | +3 |

Chi tiết từng phần: [`BONUS-CHALLENGE.md`](BONUS-CHALLENGE.md).

---

## 3. Viết bài phản tư

Mở [`submission/REFLECTION.md`](submission/REFLECTION.md) và điền bằng **số liệu thật của bạn**:

- **§1–§2:** cấu hình đã chạy và các con số trong `adapters/dpo/dpo_metrics.json`.
- **§3 — Đọc đường reward (≥ 100 từ):** `chosen` tăng hay giảm? Margin tăng vì đâu? Held-out có đi cùng hướng
  với huấn luyện không? Chẩn đoán tự động có khớp với điều bạn thấy không?
- **§4 — So sánh SFT và SFT+DPO:** điền bảng từ `judge_summary.json`, rồi chọn 2 ví dụ cụ thể (1 về độ hữu ích,
  1 về an toàn) để giải thích.
- **§6 — Một quyết định quan trọng (≥ 150 từ):** chọn một lựa chọn (β, tốc độ học, lượng dữ liệu, giám khảo…),
  nêu phương án thay thế, lý do chọn, kết quả và điều bạn sẽ đổi nếu làm lại.

Các mục bonus (§5, §7–§9) chỉ cần điền nếu bạn làm phần tương ứng.

---

## 4. Nộp bài

1. Tạo repo **public** trên GitHub của bạn và đưa toàn bộ thư mục lab lên.
2. Commit:
   - các notebook NB0–NB4 **còn giữ output** (hoặc file Colab đã chạy xong),
   - các ảnh trong `submission/screenshots/`: `02-sft-loss.png`, `02b-pref-length.png`,
     `03-dpo-reward-curves.png`, `04-side-by-side-table.png`,
   - các file kết quả đã tải về ở mục 1 (`data/eval/`, các file `.json` của `adapters/dpo/`),
   - `submission/REFLECTION.md` đã điền.
3. Nếu chạy trên máy riêng, chạy `make verify` để tự kiểm tra; lệnh phải kết thúc không lỗi.
4. Nộp đường dẫn repo vào LMS. Giữ repo public đến khi có điểm.

Lưu ý: bài được chấm từ repo nên **chỉ file đã commit mới được tính**. File `.gitignore` đã chặn trọng số mô
hình; tuyệt đối không commit file `.env` hay khoá API.

**Hạn nộp:** 23:59 ngày hôm sau. Nộp muộn trừ 10% mỗi ngày, quá 3 ngày được 0 điểm. Phúc khảo trong vòng 1 tuần.

---

## 5. Thang điểm (tóm tắt)

Lab chiếm 30% điểm Daily Lab của Track 3. Bạn được chấm theo **bằng chứng và cách bạn giải thích**, không theo
điểm số tuyệt đối của mô hình.

| Phần | Tiêu chí | Điểm |
|---|---|---:|
| NB0 | `my_dpo_loss` qua các kiểm tra | 6 |
| NB0 | Trả lời câu hỏi về dịch chuyển xác suất | 4 |
| NB1 | Loss SFT giảm; lưu được `models/sft-merged/` | 8 |
| NB2 | Chia dữ liệu theo câu hỏi, không trùng; đã đọc 3 cặp mẫu | 8 |
| NB2 | Đo thiên vị độ dài (ảnh + tỉ lệ `chosen` dài hơn) | 4 |
| NB3 | Adapter DPO huấn luyện trên `models/sft-merged` | 6 |
| NB3 | Biểu đồ reward cho cả huấn luyện **và** held-out, tách `chosen`/`rejected` | 10 |
| NB3 | Giải thích chẩn đoán trong REFLECTION §3 | 8 |
| NB4 | 8 câu cố định + ≥ 50 câu held-out được sinh câu trả lời | 6 |
| NB4 | Chấm tự động: win rate kèm CI 95%, sanity accuracy, tỉ lệ câu dài thắng, win rate các cặp dài gần bằng | 10 |
| Phản tư | §3, §4, §6 trả lời bằng số liệu của bạn | 20 |
| Tái lập | Chạy lại từ đầu được (Colab "Chạy tất cả" hoặc `make pipeline`) | 5 |
| Kiểm tra | `make verify` chạy không lỗi | 5 |
| | **Tổng phần bắt buộc** | **100** |

Chỉ thấy margin tăng thì **chưa đủ** 10 điểm biểu đồ: cần đường held-out và tách riêng `chosen`/`rejected`.
Bản đầy đủ: [`rubric.md`](rubric.md).

---

## 6. Gặp lỗi?

| Triệu chứng | Cách xử lý |
|---|---|
| Báo hết bộ nhớ GPU (`CUDA out of memory`) | Thêm dòng `os.environ["MAX_LEN"] = "512"` vào cell cài đặt đầu tiên, rồi chạy lại từ đầu. |
| NB3 chạy rất lâu | Bình thường (~40–60 phút). Nếu chỉ muốn thử luồng chạy, thêm `os.environ["PREF_TRAIN"] = "200"` vào cell cài đặt. |
| Hết giờ GPU Colab | File trong phiên bị mất. Luôn tải kết quả về trước (mục 1); phiên mới phải chạy lại từ NB1. |
| `my_dpo_loss` báo sai | So kết quả với dòng "Đáp số tham chiếu" mà notebook in ra; kiểm tra dấu trừ và hệ số β. |
| Lỗi khác | Xem mục "Lỗi thường gặp" trong [`docs/reference.md`](docs/reference.md). |

---

Mã nguồn theo giấy phép MIT ([`LICENSE`](LICENSE)). Nguồn dữ liệu, giấy phép dữ liệu, công cụ sử dụng và lời
cảm ơn: [`docs/reference.md`](docs/reference.md).

© Chương trình AICB, VinUniversity · Track 3 · Ngày 22.
