"""CPU unit tests for the lab22 package: data split, judge logic, DPO math.

Run:  pytest -q scripts/   (or `make test`). No GPU, no network.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lab22 import data as D
from lab22 import judge as J
from lab22 import math_reward as MR


def pair(prompt: str, chosen: str = "good answer", rejected: str = "bad") -> dict:
    return D.to_conversational({"prompt": prompt, "chosen": chosen, "rejected": rejected})


# --- data -------------------------------------------------------------------

def test_message_text_takes_last_assistant_turn():
    convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert D.message_text(convo) == "a"
    assert D.message_text("plain") == "plain"
    with pytest.raises(ValueError):
        D.message_text([])


def test_prompt_falls_back_to_first_user_turn():
    row = {"chosen": [{"role": "user", "content": "hỏi"}, {"role": "assistant", "content": "x"}], "rejected": "y"}
    assert D.to_conversational(row)["prompt"] == [{"role": "user", "content": "hỏi"}]


def test_split_by_prompt_is_disjoint_and_keeps_duplicates_together():
    pairs = [pair(f"prompt {i % 30}", chosen=f"c{i}") for i in range(90)]
    train, held = D.split_by_prompt(pairs, n_train=40, n_eval=12, seed=1)
    D.assert_disjoint(train, held)
    assert len(train) == 40 and len(held) == 12
    # Case/whitespace variants of a train prompt count as leakage.
    leaked = [pair("  PROMPT   " + train[0]["prompt"][0]["content"].split()[-1])]
    with pytest.raises(AssertionError):
        D.assert_disjoint(train, leaked)


def test_split_is_deterministic():
    pairs = [pair(f"p{i}") for i in range(50)]
    assert D.split_by_prompt(pairs, 20, 5, seed=3) == D.split_by_prompt(pairs, 20, 5, seed=3)


def test_unusable_pairs_are_rejected():
    assert not D.is_usable_pair(pair("q?", chosen="same", rejected="same"))
    assert D.is_usable_pair(pair("q?"))


def test_length_stats():
    rows = [pair("a", "long answer", "s"), pair("b", "s", "longer one")]
    stats = D.length_stats(rows)
    assert stats["n"] == 2 and stats["chosen_longer_frac"] == 0.5


# --- judge ------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"winner": "A", "reason": "x"}', "A"),
        ('noise {"winner":"b"} tail', "B"),
        ("tie", "tie"),
        ("???", "invalid"),
        ("{not json}", "invalid"),
    ],
)
def test_parse_verdict(raw, expected):
    assert J.parse_verdict(raw) == expected


@pytest.mark.parametrize(
    "first,second,winner,consistent",
    [
        ("B", "A", "dpo", True),   # DPO preferred in both orders
        ("A", "B", "sft", True),   # SFT preferred in both orders
        ("A", "A", "tie", False),  # always picks slot A: position bias
        ("B", "B", "tie", False),
        ("tie", "tie", "tie", True),
        ("B", "tie", "tie", False),
    ],
)
def test_judge_pair_combines_both_orders(first, second, winner, consistent):
    replies = iter([first, second])
    out = J.judge_pair("p", "sft text", "dpo text", lambda s, u: next(replies))
    assert out["winner"] == winner
    assert out["position_consistent"] is consistent


def test_judge_pair_swaps_answers_between_calls():
    seen = []
    J.judge_pair("p", "SFT-ANSWER", "DPO-ANSWER", lambda s, u: seen.append(u) or "tie")
    assert seen[0].index("SFT-ANSWER") < seen[0].index("DPO-ANSWER")
    assert seen[1].index("DPO-ANSWER") < seen[1].index("SFT-ANSWER")


def test_invalid_verdict_is_retried_then_marked_failed():
    replies = iter(["garbage", '{"winner": "B"}', "x", "y"])  # 1st order recovers, 2nd stays invalid
    out = J.judge_pair("p", "s", "d", lambda s, u: next(replies))
    assert out["order_sft_first"] == "B" and out["order_dpo_first"] == "invalid"
    assert out["winner"] == "failed" and out["position_consistent"] is None


def test_summarize_excludes_and_counts_failed_pairs():
    recs = [
        {"winner": "dpo", "sft": "s", "dpo": "dd", "position_consistent": True},
        {"winner": "failed", "sft": "s", "dpo": "dd", "position_consistent": None},
    ]
    s = J.summarize(recs)
    assert s["n"] == 1 and s["n_failed"] == 1 and s["dpo_win_rate"] == 1.0


def test_longer_won_compares_winner_with_loser_and_skips_equal_lengths():
    recs = [
        {"winner": "sft", "sft": "short", "dpo": "much longer", "position_consistent": True},  # shorter won
        {"winner": "dpo", "sft": "same!", "dpo": "same?", "position_consistent": True},  # equal: skipped
    ]
    assert J.summarize(recs)["longer_answer_won_frac"] == 0.0


def test_summarize_counts_ties_as_half_and_reports_length_bias():
    recs = [
        {"winner": "dpo", "sft": "s", "dpo": "longer", "position_consistent": True},
        {"winner": "dpo", "sft": "s", "dpo": "longer", "position_consistent": True},
        {"winner": "sft", "sft": "longer", "dpo": "s", "position_consistent": True},
        {"winner": "tie", "sft": "s", "dpo": "s", "position_consistent": False},
    ]
    s = J.summarize(recs, seed=0)
    assert s["n"] == 4 and s["ties"] == 1
    assert s["dpo_win_rate"] == pytest.approx(2.5 / 4)
    assert s["longer_answer_won_frac"] == 1.0
    assert s["position_consistency"] == 0.75
    low, high = s["win_rate_ci95"]
    assert 0.0 <= low <= s["dpo_win_rate"] <= high <= 1.0


def test_summarize_without_records_is_not_judged():
    assert J.summarize([]) == {"n": 0, "n_failed": 0, "status": "not judged"}


def test_panel_record_needs_unanimous_judges():
    dpo, sft, tie, failed = ({"winner": w} for w in ("dpo", "sft", "tie", "failed"))
    assert J.panel_record([dpo, dpo])["winner"] == "dpo"
    assert J.panel_record([dpo, sft])["winner"] == "tie"
    assert J.panel_record([dpo, tie])["winner"] == "tie"
    assert J.panel_record([dpo, failed])["winner"] == "dpo"
    assert J.panel_record([failed, failed])["winner"] == "failed"


def test_rm_record_picks_higher_score_and_has_no_position():
    assert J.rm_record(0.2, 1.5)["winner"] == "dpo"
    assert J.rm_record(1.5, 0.2)["winner"] == "sft"
    assert J.rm_record(1.0, 1.0)["winner"] == "tie"
    assert J.rm_record(0.0, 1.0)["position_consistent"] is None


def test_summarize_reward_model_records():
    recs = [
        {"id": "a", "winner": "dpo", "sft": "s", "dpo": "longer", "sft_score": 0.0, "dpo_score": 2.0,
         "position_consistent": None},
        {"id": "b", "winner": "sft", "sft": "abcd", "dpo": "abce", "sft_score": 1.0, "dpo_score": 0.5,
         "position_consistent": None},
    ]
    s = J.summarize(recs)
    assert s["position_consistency"] is None  # undefined without A/B orders
    assert s["length_matched_n"] == 1 and s["length_matched_win_rate"] == 0.0
    assert -1.0 <= s["score_length_spearman"] <= 1.0


def test_spearman_handles_ties_and_constant_input():
    assert J.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert J.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert J.spearman([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1.0)
    assert J.spearman([1, 1, 1], [1, 2, 3]) is None


def test_agreement_skips_failed_and_unshared_prompts():
    a = [{"id": "1", "winner": "dpo"}, {"id": "2", "winner": "sft"}, {"id": "3", "winner": "failed"}]
    b = [{"id": "1", "winner": "dpo"}, {"id": "2", "winner": "tie"}, {"id": "3", "winner": "dpo"}]
    assert J.agreement(a, b) == {"n": 2, "agreement": 0.5}


def test_sanity_set_catches_a_length_only_judge():
    good = {g for _, g, _ in J.SANITY_PAIRS}
    assert J.sanity_accuracy(lambda p, a: 1.0 if a in good else 0.0) == 1.0
    length_only = J.sanity_accuracy(lambda p, a: float(len(a)))
    assert length_only < 0.8, length_only  # the sanity bar must reject a judge that only rewards length


def test_make_caller_requires_model_id(monkeypatch):
    with pytest.raises(RuntimeError, match="JUDGE_MODEL"):
        J.make_caller("openai", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        J.make_caller("openai", "some-model")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        J.make_caller("gemini", "some-model")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        J.make_caller("openrouter", "google/gemini-2.5-flash")
    with pytest.raises(RuntimeError, match="JUDGE_PROVIDER"):
        J.make_caller("other", "m")


class _CharTokenizer:
    """One token per character; enough to test the token budget filter."""

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return "".join(m["content"] for m in messages) + ("T" if kwargs.get("enable_thinking") is False else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}


def test_fits_budget_checks_both_sides_in_tokens():
    tok = _CharTokenizer()
    row = pair("q", chosen="ab", rejected="abcdefgh")  # chosen short, rejected long
    assert D.fits_budget(row, tok, max_len=9)
    assert not D.fits_budget(row, tok, max_len=8)
    assert not D.fits_budget(row, tok, max_len=9, template_kwargs={"enable_thinking": False})


def test_split_mismatch_detects_regenerated_split(tmp_path):
    pref, adapter = tmp_path / "pref", tmp_path / "adapter"
    pref.mkdir()
    adapter.mkdir()
    for name in ("train", "eval"):
        (pref / f"{name}.parquet").write_bytes(name.encode())
    assert "missing" in D.split_mismatch(pref, adapter)
    D.save_split_fingerprint(pref, adapter)
    assert D.split_mismatch(pref, adapter) is None
    (pref / "eval.parquet").write_bytes(b"reshuffled")
    assert "changed" in D.split_mismatch(pref, adapter)


# --- DPO math (torch on CPU) ------------------------------------------------

torch = pytest.importorskip("torch")
from lab22 import dpo_math as M


def test_dpo_loss_is_log2_when_policy_equals_reference():
    logps = torch.tensor([-12.0, -30.0])
    loss, rc, rr = M.dpo_loss(logps, logps + 1, logps, logps + 1, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert torch.all(rc == 0) and torch.all(rr == 0)


def test_dpo_loss_matches_closed_form():
    t = torch.tensor
    loss, rc, rr = M.dpo_loss(t([-10.0]), t([-20.0]), t([-12.0]), t([-15.0]), beta=0.5)
    assert rc.item() == pytest.approx(1.0) and rr.item() == pytest.approx(-2.5)
    assert loss.item() == pytest.approx(-math.log(1 / (1 + math.exp(-3.5))), rel=1e-5)


def test_sequence_logps_masks_prompt_tokens():
    logits = torch.zeros(1, 4, 5)  # uniform → each token log p = -log 5
    labels = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[0, 0, 1, 1]])
    total, mean = M.sequence_logps(logits, labels, mask)
    assert total.item() == pytest.approx(-2 * math.log(5), rel=1e-5)
    assert mean.item() == pytest.approx(-math.log(5), rel=1e-5)


def test_ipo_is_length_normalized():
    t = torch.tensor
    # Same per-token log-ratio margin, different lengths: the loss must not change.
    short = M.ipo_loss(t([-1.0]), t([-2.0]), t([-2.0]), t([-1.0]), 10, 10, beta=0.1)
    long = M.ipo_loss(t([-10.0]), t([-20.0]), t([-20.0]), t([-10.0]), 100, 100, beta=0.1)
    assert short.item() == pytest.approx(long.item())
    assert short.item() == pytest.approx((0.2 - 1 / (2 * 0.1)) ** 2)


def test_rpo_adds_chosen_nll_and_orpo_has_no_reference():
    pc, pr = torch.tensor([-5.0]), torch.tensor([-9.0])
    base, _, _ = M.dpo_loss(pc, pr, pc, pr)
    nll = torch.tensor([2.0])
    assert M.rpo_loss(pc, pr, pc, pr, nll, alpha=0.5).item() == pytest.approx(base.item() + 1.0)
    assert M.orpo_loss(torch.tensor([-0.5]), torch.tensor([-2.0]), nll).item() > nll.item()
    good = M.simpo_loss(torch.tensor([-0.5]), torch.tensor([-2.0])).item()
    bad = M.simpo_loss(torch.tensor([-2.0]), torch.tensor([-0.5])).item()
    assert good < bad


# --- reward diagnosis (pandas) ----------------------------------------------

pd = pytest.importorskip("pandas")
from lab22 import modeling as MD


@pytest.mark.parametrize(
    "chosen,rejected,label",
    [
        ([0.0, 0.2, 0.4], [0.0, -0.2, -0.4], "INTENDED"),
        ([0.0, -0.2, -0.4], [0.0, -0.6, -1.2], "LIKELIHOOD DISPLACEMENT"),
        ([0.0, -0.1, -0.2], [0.0, 0.1, 0.2], "FAILURE"),
    ],
)
def test_diagnose(chosen, rejected, label):
    df = pd.DataFrame({"step": [1, 2, 3], "rewards/chosen": chosen, "rewards/rejected": rejected})
    assert MD.diagnose(df, window=1)[0] == label


def test_reward_history_strips_eval_prefix():
    log = [{"step": 5, "loss": 0.6}, {"step": 25, "eval_rewards/chosen": 0.1, "eval_rewards/rejected": -0.1}]
    df = MD.reward_history(log, prefix="eval_")
    assert list(df["rewards/chosen"]) == [0.1]


# --- GRPO math reward -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ref", "ok"),
    [
        ("Vậy\nĐáp số: 1.440", "1440", True),  # Vietnamese thousands grouping
        ("Đáp số: 1,440", "1440", True),  # English thousands grouping
        ("Đáp số: 2,5 kg", "2.5", True),  # Vietnamese decimal comma
        ("Đáp số: 2.5.", "2.5", True),
        ("Đáp số: 1.234,5", "1234.5", True),
        ("Đáp số: 25", "2.5", False),
        ("Đáp số: 125", "0.125", False),  # canonical reference is not regrouped
        ("Bước 1: 3 + 4 = 7. Đáp số: 8", "7", False),  # only the final marker counts
        ("không có số", "3", False),
    ],
)
def test_math_reward_reads_vietnamese_numbers(text, ref, ok):
    assert MR.is_correct(MR.extract_answer(text), ref) is ok
