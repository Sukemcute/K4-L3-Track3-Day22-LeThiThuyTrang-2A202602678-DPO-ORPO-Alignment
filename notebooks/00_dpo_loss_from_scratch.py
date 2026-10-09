# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB0 — DPO loss tự cài từ đầu (CPU, ~10 phút)
#
# **Không cần GPU.** Trước khi gọi `DPOTrainer`, bạn tự viết loss và kiểm tra nó
# trên số liệu đồ chơi. Phần này lấy từ lab K3 (tự cài DPO) và là nền để đọc
# đường cong reward ở NB3.
#
# Bạn sẽ thấy:
# 1. Tại bước 0 (mô hình đang học (policy) = reference) loss luôn bằng `log 2 ≈ 0.693`.
# 2. Gradient của DPO bị nhân với `sigmoid(-margin)`: cặp đã phân biệt tốt gần như không còn được học.
# 3. **Likelihood displacement**: loss vẫn giảm khi log-prob của *chosen* giảm, miễn rejected giảm nhanh hơn.
# 4. IPO, RPO, SimPO, ORPO khác DPO ở đâu, trên cùng một bộ số.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

import math

import torch
import torch.nn.functional as F

from lab22 import dpo_math as M

torch.manual_seed(0)

# %% [markdown]
# ## 1. Log-prob của một câu trả lời
#
# `log π(y|x) = Σ_t log π(y_t | x, y_<t)`, chỉ cộng trên token của câu trả lời
# (mask = 1), không cộng trên câu hỏi.

# %%
vocab, length = 8, 5
logits = torch.randn(1, length, vocab)
labels = torch.randint(0, vocab, (1, length))
mask = torch.tensor([[0, 0, 1, 1, 1]])  # 2 token prompt, 3 token trả lời
total, mean = M.sequence_logps(logits, labels, mask)
print(f"sum log p = {total.item():.3f}   mean log p = {mean.item():.3f}")

# %% [markdown]
# ## 2. Bài tập: tự viết DPO loss
#
# Công thức (Rafailov et al. 2023):
#
# $$\mathcal{L} = -\log\sigma\Big(\beta\big[(\log\pi_\theta(y_w) - \log\pi_{ref}(y_w)) - (\log\pi_\theta(y_l) - \log\pi_{ref}(y_l))\big]\Big)$$
#
# Hàm dưới đây giữ nguyên DPO sigmoid của đề bài. Ô kiểm tra sẽ so với bản
# tham chiếu trong `lab22/dpo_math.py`, kiểm tra gradient và chạy một bước học.
# Log-prob đầu vào phải là tổng trên token completion, đã mask prompt/padding.
# Tính log-prob reference trong `torch.no_grad()` và giữ reference cố định.


# %%
def my_dpo_loss(pc, pr, rc, rr, beta=0.1):
    """Mean sigmoid DPO loss for paired completion log-probabilities.

    Inputs are floating tensors of identical shape/device (one value per
    response). Gradients flow to pc/pr only. FP16/BF16 inputs are promoted
    before subtraction; FP64 is preserved for gradient checks.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    inputs = (pc, pr, rc, rr)
    if any(not t.is_floating_point() for t in inputs):
        raise TypeError("log-probabilities must be floating-point tensors")
    if pc.numel() == 0 or any(t.shape != pc.shape for t in inputs):
        raise ValueError("log-probabilities must have the same non-empty shape")
    if any(t.device != pc.device for t in inputs):
        raise ValueError("log-probabilities must be on the same device")

    dtype = torch.float64 if any(t.dtype == torch.float64 for t in inputs) else torch.float32
    chosen_ratio = pc.to(dtype) - rc.detach().to(dtype)
    rejected_ratio = pr.to(dtype) - rr.detach().to(dtype)
    # logsigmoid avoids underflow from log(sigmoid(margin)) on difficult pairs.
    margin = beta * (chosen_ratio - rejected_ratio)
    return -F.logsigmoid(margin).mean()


# %%
pc, pr = torch.tensor([-12.0, -30.0]), torch.tensor([-15.0, -28.0])
rc, rr = torch.tensor([-13.0, -29.0]), torch.tensor([-14.0, -29.0])
ref_loss, _, _ = M.dpo_loss(pc, pr, rc, rr, beta=0.1)
mine = my_dpo_loss(pc, pr, rc, rr, beta=0.1)
assert torch.allclose(mine, ref_loss, atol=1e-6), (mine, ref_loss)
print(f"✓ Khớp tham chiếu: {mine.item():.4f}")

# %% [markdown]
# ## 3. Bước 0: mô hình đang học (policy) = reference ⇒ loss = log 2
#
# NB3 khởi tạo mô hình đang học (policy) bằng chính mô hình SFT (LoRA mới có trọng số B = 0), nên
# reward ngầm định ban đầu bằng 0 và loss bắt đầu ở 0.693. Nếu log của bạn
# không bắt đầu gần 0.693, reference đang không phải mô hình SFT.

# %%
same = torch.tensor([-20.0, -35.0])
loss0, cr0, rr0 = M.dpo_loss(same, same - 3, same, same - 3)
mine0 = my_dpo_loss(same, same - 3, same, same - 3)
assert torch.allclose(mine0, torch.tensor(math.log(2)), atol=1e-6)
assert torch.equal(cr0, torch.zeros_like(cr0)) and torch.equal(rr0, torch.zeros_like(rr0))
print(f"loss at init = {loss0.item():.4f}   log 2 = {math.log(2):.4f}   rewards = {cr0.tolist()}, {rr0.tolist()}")

# %% [markdown]
# ## 4. Trọng số gradient = sigmoid(−margin)
#
# Với batch N cặp và margin đã nhân β, gradient theo `pc` là
# `−β sigmoid(−margin)/N`, theo `pr` là dấu ngược lại. Gradient descent ưu tiên
# chosen; cặp khó có trọng số lớn hơn cặp đã phân biệt tốt. β điều chỉnh cả
# độ dốc lẫn mức bão hoà, nên cần xem cùng learning rate và reward held-out.

# %%
for margin in (-2.0, 0.0, 2.0, 5.0):
    m = torch.tensor(margin, requires_grad=True)
    loss = -torch.nn.functional.logsigmoid(m)
    loss.backward()
    print(f"margin {margin:+.1f}: loss {loss.item():.3f}   |dL/dmargin| {abs(m.grad.item()):.3f}")

# %% [markdown]
# ### Kiểm tra gradient, độ ổn định số và một vòng tối ưu
#
# Các kiểm tra dùng chính `my_dpo_loss`. Mô hình đồ chơi softmax có hai câu
# trả lời; đây là kiểm tra cơ chế tối ưu, chưa phải fine-tune Qwen ở NB3.

# %%
pc_g = pc.clone().requires_grad_()
pr_g = pr.clone().requires_grad_()
rc_g = rc.clone().requires_grad_()
rr_g = rr.clone().requires_grad_()
my_dpo_loss(pc_g, pr_g, rc_g, rr_g).backward()
margin_g = 0.1 * ((pc - rc) - (pr - rr))
expected_grad = 0.1 * torch.sigmoid(-margin_g) / pc.numel()
assert torch.allclose(pc_g.grad, -expected_grad)
assert torch.allclose(pr_g.grad, expected_grad)
assert rc_g.grad is None and rr_g.grad is None
double_inputs = tuple(t.double().requires_grad_() for t in (pc, pr))
assert torch.autograd.gradcheck(
    lambda c, r: my_dpo_loss(c, r, rc.double(), rr.double()), double_inputs
)
print("✓ Gradient đúng công thức và finite differences; reference không nhận gradient")

for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
    hard_c = torch.tensor([-60000.0, 0.0], dtype=dtype, requires_grad=True)
    hard_r = torch.tensor([0.0, -60000.0], dtype=dtype, requires_grad=True)
    ref_extreme = torch.full_like(hard_c, -30000.0)
    stable_loss = my_dpo_loss(hard_c, hard_r, ref_extreme, ref_extreme, beta=1.0)
    stable_loss.backward()
    assert torch.isfinite(stable_loss) and stable_loss >= 0
    assert torch.isfinite(hard_c.grad).all() and torch.isfinite(hard_r.grad).all()
    # BF16 rounds 60000 to 59904; compare against the represented inputs.
    expected_loss, _, _ = M.dpo_loss(
        hard_c.detach().double(), hard_r.detach().double(),
        ref_extreme.double(), ref_extreme.double(), beta=1.0,
    )
    assert torch.allclose(stable_loss.double(), expected_loss)
print("✓ Loss/gradient hữu hạn ở margin khoảng ±60000, với FP16/BF16/FP32/FP64")

# Reject accidental broadcasting and meaningless beta instead of silently training.
for bad_inputs, bad_beta in (((pc, pr[:1], rc, rr), 0.1), ((pc, pr, rc, rr), 0.0)):
    try:
        my_dpo_loss(*bad_inputs, beta=bad_beta)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid DPO inputs should be rejected")

toy_logits = torch.nn.Parameter(torch.zeros(2))
optimizer = torch.optim.SGD([toy_logits], lr=0.5)
toy_ref = torch.log_softmax(toy_logits.detach(), dim=0)
initial_loss = my_dpo_loss(toy_ref[:1], toy_ref[1:], toy_ref[:1], toy_ref[1:]).item()
for _ in range(40):
    optimizer.zero_grad()
    toy_logps = torch.log_softmax(toy_logits, dim=0)
    toy_loss = my_dpo_loss(toy_logps[:1], toy_logps[1:], toy_ref[:1], toy_ref[1:])
    toy_loss.backward()
    optimizer.step()
final_logps = torch.log_softmax(toy_logits.detach(), dim=0)
final_loss = my_dpo_loss(final_logps[:1], final_logps[1:], toy_ref[:1], toy_ref[1:]).item()
assert final_loss < initial_loss
assert final_logps[0] > toy_ref[0] and final_logps[1] < toy_ref[1]
print(f"✓ 40 bước SGD: loss {initial_loss:.4f} → {final_loss:.4f}; "
      f"P(chosen) 0.5000 → {final_logps[0].exp().item():.4f}")

# %% [markdown]
# ## 5. Likelihood displacement bằng số
#
# Hai kịch bản đều làm margin tăng 2 nat. Loss giống hệt nhau, nhưng ở kịch
# bản B log-prob của câu *được chọn* lại giảm. DPO không phân biệt được hai
# trường hợp này; chỉ đường cong `rewards/chosen` ở NB3 cho bạn biết.

# %%
ref_c, ref_r = torch.tensor([-20.0]), torch.tensor([-22.0])
scenarios = {
    "A: chosen ↑, rejected ↓": (ref_c + 1, ref_r - 1),
    "B: chosen ↓, rejected ↓↓": (ref_c - 3, ref_r - 5),
}
for name, (pc_, pr_) in scenarios.items():
    _, cr, rj = M.dpo_loss(pc_, pr_, ref_c, ref_r, beta=1.0)
    loss = my_dpo_loss(pc_, pr_, ref_c, ref_r, beta=1.0)
    print(f"{name:28s} loss {loss.item():.3f}  reward chosen {cr.item():+.1f}  rejected {rj.item():+.1f}")
loss_a, loss_b = [my_dpo_loss(c, r, ref_c, ref_r, beta=1.0) for c, r in scenarios.values()]
assert torch.allclose(loss_a, loss_b)

# %% [markdown]
# **RPO** thêm NLL của câu chosen vào loss: kịch bản B bị phạt vì chosen bị đẩy xuống.
#
# **Trả lời NB0:** DPO tối ưu chênh lệch log-ratio với reference, nên không
# bảo đảm log-prob chosen tăng riêng lẻ. Ở A, chosen tăng 1 và rejected giảm 1;
# ở B, chosen giảm 3 nhưng rejected giảm 5. Cả hai đều có margin +2 và loss
# `−log sigmoid(2) ≈ 0.127`, thấp hơn loss ban đầu 0.693. Vì thế cần quan sát
# riêng chosen/rejected trên train và held-out ở NB3. RPO bổ sung mục tiêu
# bắt chước chosen để hạn chế dịch chuyển này; cần đánh giá thực nghiệm.

# %%
for name, (pc_, pr_) in scenarios.items():
    nll = -pc_ / 10  # NLL trung bình trên 10 token
    print(f"{name:28s} RPO loss {M.rpo_loss(pc_, pr_, ref_c, ref_r, nll, beta=1.0).item():.3f}")

# %% [markdown]
# ## 6. Bốn biến thể trên cùng một cặp
#
# | Loss | Cần mô hình tham chiếu (reference)? | Chuẩn hoá độ dài? | Ghi chú |
# |---|---|---|---|
# | DPO (sigmoid) | có | không | mức cơ sở (baseline) |
# | IPO | có | có (TRL chia theo số token) | hồi quy margin về 1/(2β), chống quá khớp khi dữ liệu gần như tất định |
# | RPO | có | không | DPO + NLL(chosen), giảm likelihood displacement |
# | SimPO | không | có | log-prob trung bình + margin γ |
# | ORPO | không | có | NLL(chosen) + λ·log-odds-ratio, gộp SFT và sở thích vào một bước |
#
# NB3b huấn luyện thật các biến thể này (TRL `loss_type` và `trl.experimental.orpo`).

# %%
n_tokens_c, n_tokens_r = 40, 120  # chosen ngắn, rejected dài
pc_, pr_ = torch.tensor([-48.0]), torch.tensor([-130.0])
rc_, rr_ = torch.tensor([-50.0]), torch.tensor([-128.0])
avg_c, avg_r = pc_ / n_tokens_c, pr_ / n_tokens_r
print(f"DPO   {M.dpo_loss(pc_, pr_, rc_, rr_)[0].item():.4f}")
print(f"IPO   {M.ipo_loss(pc_, pr_, rc_, rr_, n_tokens_c, n_tokens_r).item():.4f}")
print(f"SimPO {M.simpo_loss(avg_c, avg_r).item():.4f}")
print(f"ORPO  {M.orpo_loss(avg_c, avg_r, -avg_c).item():.4f}")

# %% [markdown]
# **Câu hỏi cho REFLECTION §3:** tổng log-prob thường âm hơn khi có nhiều token
# với mức log-prob trung bình tương tự.
# Vì sao điều đó khiến DPO gốc dễ thiên vị độ dài, và SimPO/ORPO xử lý bằng cách nào?
#
# **Trả lời:** DPO dùng tổng log-prob; số token ảnh hưởng độ lớn log-ratio và
# gradient. Nếu chosen thường dài hơn, dữ liệu có thể khuyến khích viết dài.
# DPO không mặc định ưu tiên câu dài trong mọi cặp: còn phụ thuộc policy,
# reference và nhãn sở thích. SimPO dùng log-prob trung bình mỗi token; ORPO
# dùng xác suất từ log-prob trung bình để tính odds và thêm NLL chosen.
# Chuẩn hoá giảm ảnh hưởng trực tiếp của số token, nhưng không xoá thiên vị
# trong nhãn dữ liệu. NB2/NB4 phải đo length bias và đánh giá các cặp gần bằng
# độ dài. NB3 dùng DPOTrainer của TRL; hàm NB0 minh hoạ loss và autograd, còn
# hiệu quả fine-tune thật cần được kiểm chứng bằng reward/đánh giá held-out.
