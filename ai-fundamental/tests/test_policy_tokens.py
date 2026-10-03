"""Policy gradients must score the tokens that were actually sampled."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from fundamental import agent
from fundamental.posttrain import sample_completion, sampled_token_stats
from fundamental.tokenizer import ByteBPE, EOS


class FixedLogits(nn.Module):
    def __init__(self, vocab_size=259):
        super().__init__()
        self.cfg = SimpleNamespace(max_seq_len=16)
        self.bias = nn.Parameter(torch.linspace(-1, 1, vocab_size))

    def forward(self, ids):
        return self.bias.expand(ids.shape[0], ids.shape[1], -1)


def test_policy_logprob_uses_sampled_ids_without_implicit_eos():
    tok = ByteBPE([(ord("a"), ord("b"))])
    sampled = [ord("a"), ord("b")]
    assert tok.decode(sampled) == "ab"
    assert tok.encode("ab") == [258]  # Text roundtrip changes token IDs.
    model = FixedLogits()
    logp, count, _ = sampled_token_stats(model, tok.encode("Q"), sampled, "cpu")
    expected = F.log_softmax(model.bias, dim=0)[sampled].sum()
    assert count == 2
    torch.testing.assert_close(logp, expected)
    assert logp.item() != F.log_softmax(model.bias, dim=0)[[258, EOS]].sum().item()
    (-logp).backward()
    assert model.bias.grad is not None


def test_sampled_eos_is_included_only_when_drawn():
    tok = ByteBPE()
    model = FixedLogits(vocab_size=258)
    with torch.no_grad():
        model.bias.fill_(-1e9)
        model.bias[EOS] = 0
    text, sampled = sample_completion(model, tok, "Q", "cpu", return_tokens=True)
    assert text == ""
    assert sampled == [EOS]
    logp, count, _ = sampled_token_stats(model, tok.encode("Q"), sampled, "cpu")
    assert count == 1
    assert torch.isfinite(logp)


def test_self_judge_control_has_matched_candidate_budget(monkeypatch):
    counter = {"n": 0}

    def propose(*args, **kwargs):
        counter["n"] += 1
        n = counter["n"]
        return f"candidate-{n}", [n]

    monkeypatch.setattr(agent, "sample_completion", propose)
    monkeypatch.setattr(agent, "self_judge_score", lambda *args: int(args[3].split("-")[-1]))
    baseline, base_details = agent.choose_action(None, None, "observation", "cpu", False, 3, 2)
    counter["n"] = 0
    judged, judge_details = agent.choose_action(None, None, "observation", "cpu", True, 3, 2)
    assert baseline == "candidate-1"
    assert judged == "candidate-3"
    assert base_details["candidate_count"] == judge_details["candidate_count"] == 3
    assert base_details["sampled_tokens"] == judge_details["sampled_tokens"] == 3
