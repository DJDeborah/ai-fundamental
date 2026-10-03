import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)
pytest.importorskip("triton")

from fundamental.triton_attention import causal_attention


@pytest.mark.parametrize("length", [33, 128, 513])
@pytest.mark.parametrize("head_dim", [32, 64])
def test_triton_matches_sdpa(length, head_dim):
    q, k, v = [torch.randn(2, 3, length, head_dim, device="cuda", dtype=torch.float16) for _ in range(3)]
    actual = causal_attention(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
