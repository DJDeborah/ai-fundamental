import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "nn"):
    pytest.skip("PyTorch unavailable in this Python installation", allow_module_level=True)

from fundamental.model import GPT, config_for
from fundamental.posttrain import verified_answer


def test_model_backward_and_causality():
    torch.manual_seed(1)
    cfg = config_for("tiny", 300)
    model = GPT(cfg).eval()
    a = torch.randint(0, 300, (1, 12))
    b = a.clone()
    b[0, 8:] = torch.randint(0, 300, (4,))
    with torch.no_grad():
        la = model(a)
        lb = model(b)
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)
    _, loss = model(a[:, :-1], a[:, 1:])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embed.weight.grad is not None


def test_base100m_parameter_count():
    with torch.device("meta"):
        model = GPT(config_for("base100m", 8192))
    assert 99_000_000 < model.count_parameters() < 103_000_000


def test_exact_verifier():
    assert verified_answer("42 ", "42") == 1
    assert verified_answer("The answer is 42", "42") == 0
