"""
test_hadamard.py — CPU unit tests for Walsh-Hadamard Transform.

Run: pytest tests/unit/test_hadamard.py -v
No GPU required.
"""
import pytest
import torch
from custom_kv.ops_cpu import hadamard_rotate_cpu

SEED = 42
D = 128


@pytest.fixture
def rng():
    torch.manual_seed(SEED)
    return torch


def test_hadamard_output_shape(rng):
    """Output shape must match input shape."""
    for shape in [(D,), (4, D), (2, 8, D), (1, 32, 10, D)]:
        x = torch.randn(*shape)
        out = hadamard_rotate_cpu(x)
        assert out.shape == x.shape, f"Shape mismatch for input {shape}"


def test_hadamard_self_inverse(rng):
    """WHT is its own inverse: H(H(x)) == x."""
    x = torch.randn(D)
    x_double = hadamard_rotate_cpu(hadamard_rotate_cpu(x))
    torch.testing.assert_close(x.float(), x_double, atol=1e-5, rtol=0,
                               msg="WHT should be self-inverse")


def test_hadamard_orthogonality_matrix(rng):
    """H_n @ H_n^T == I (orthogonality)."""
    # Build H_n by rotating standard basis vectors
    H = torch.zeros(D, D)
    for i in range(D):
        e_i = torch.zeros(D)
        e_i[i] = 1.0
        H[i] = hadamard_rotate_cpu(e_i)

    # H @ H^T should be identity
    HHt = H @ H.T
    I = torch.eye(D)
    torch.testing.assert_close(HHt, I, atol=1e-5, rtol=0,
                               msg="H_n @ H_n^T must equal identity")


def test_inner_product_preserved_1d(rng):
    """
    Attention score invariance: q·k == q_rot·k_rot.
    This is the mathematical guarantee that ECC doesn't change attention behavior.
    """
    torch.manual_seed(SEED)
    for trial in range(20):
        q = torch.randn(D) * 3.0  # heavy-tailed to stress test
        k = torch.randn(D) * 3.0

        q_rot = hadamard_rotate_cpu(q)
        k_rot = hadamard_rotate_cpu(k)

        score_orig = torch.dot(q, k).item()
        score_rot  = torch.dot(q_rot, k_rot).item()

        rel_err = abs(score_orig - score_rot) / (abs(score_orig) + 1e-8)
        assert rel_err < 1e-4, (
            f"Trial {trial}: inner product not preserved. "
            f"orig={score_orig:.6f}, rot={score_rot:.6f}, rel_err={rel_err:.2e}"
        )


def test_inner_product_preserved_batched(rng):
    """Inner product preservation for batched inputs (multiple heads)."""
    torch.manual_seed(SEED)
    B, H, L = 2, 8, 10

    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)

    # Reshape to [N, D], rotate, reshape back
    N = B * H * L
    q_flat = q.reshape(N, D)
    k_flat = k.reshape(N, D)

    q_rot = hadamard_rotate_cpu(q_flat)
    k_rot = hadamard_rotate_cpu(k_flat)

    # Compute dot products for each (b, h, l) triple
    for i in range(N):
        orig = torch.dot(q_flat[i], k_flat[i]).item()
        rot  = torch.dot(q_rot[i], k_rot[i]).item()
        rel_err = abs(orig - rot) / (abs(orig) + 1e-8)
        assert rel_err < 1e-4, f"Batched index {i}: rel_err={rel_err:.2e}"


def test_hadamard_diffuses_outliers(rng):
    """
    After WHT, max activation should be smaller than before (outlier diffusion).
    This is the core motivation: spreading energy across dimensions.
    """
    torch.manual_seed(SEED)
    # Simulate pathological KV activation with large outliers
    x = torch.randn(D) * 0.5
    x[0] = 50.0   # massive outlier (typical in LLM KV activations)
    x[1] = -48.0

    x_rot = hadamard_rotate_cpu(x)

    max_orig = x.abs().max().item()
    max_rot  = x_rot.abs().max().item()

    assert max_rot < max_orig, (
        f"WHT should reduce max activation. "
        f"Before: {max_orig:.2f}, After: {max_rot:.2f}"
    )
    # After WHT, should be more uniform (max << original outlier)
    assert max_rot < max_orig / 3, (
        f"Outlier not sufficiently diffused. "
        f"max_rot={max_rot:.2f} should be < {max_orig/3:.2f}"
    )


def test_hadamard_power_of_2_requirement():
    """Should raise AssertionError for non-power-of-2 D."""
    x = torch.randn(100)  # 100 is not a power of 2
    with pytest.raises(AssertionError, match="power of 2"):
        hadamard_rotate_cpu(x)


def test_hadamard_d64(rng):
    """Works for D=64 (half head-dim models)."""
    x = torch.randn(64)
    x_double = hadamard_rotate_cpu(hadamard_rotate_cpu(x))
    torch.testing.assert_close(x.float(), x_double, atol=1e-5, rtol=0)


def test_hadamard_d256(rng):
    """Works for D=256 (larger head-dim models)."""
    x = torch.randn(256)
    x_double = hadamard_rotate_cpu(hadamard_rotate_cpu(x))
    torch.testing.assert_close(x.float(), x_double, atol=1e-5, rtol=0)
