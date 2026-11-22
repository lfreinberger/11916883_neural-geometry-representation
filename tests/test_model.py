import pytest
import torch

from neural_geometry.fit import clamped_l1, fit
from neural_geometry.model import LatentSDF, bspline_basis, hat, mlp
from neural_geometry.shapes import ellipse, primitive_scenes, sample_dataset


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_basis_is_partition_of_unity(degree):
    x = torch.linspace(-1, 1, 501)
    basis = bspline_basis(x, n_ctrl=7, degree=degree)
    assert basis.shape == (501, 7)
    assert (basis >= 0).all()
    torch.testing.assert_close(basis.sum(dim=-1), torch.ones(501))


def test_linear_basis_interpolates_the_control_points():
    nodes = torch.linspace(-1, 1, 5)
    torch.testing.assert_close(bspline_basis(nodes, n_ctrl=5, degree=1), torch.eye(5))


def test_hat_is_continuous_and_mirrored():
    x = torch.linspace(-1, 1, 4001)
    u = hat(x, n_tiles=4)
    assert u.min() == -1 and u.max() == 1
    # |du/dx| = n_tiles everywhere, so no jumps
    assert (u.diff().abs() <= 2 * 4 * (x[1] - x[0]) + 1e-6).all()
    torch.testing.assert_close(
        hat(torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]), 4),
        torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0]),
    )


def to_torch(shapes, n_points):
    points, sdf = sample_dataset(shapes, n_points, seed=0)
    return torch.from_numpy(points), torch.from_numpy(sdf)


def test_spline_reference_fits_a_circle():
    points, sdf = to_torch([ellipse(0.5, 0.5)], 2000)
    model = LatentSDF(n_ctrl=10, degree=3, code_length=1)
    codes = fit(
        model, points, sdf, steps=500, points_per_step=2000, lr_codes=0.02, delta=0.1, beta=0.0
    )
    assert clamped_l1(model(codes, points), sdf, 0.1) < 2e-3


def test_tiny_training_run_decreases_the_loss():
    torch.manual_seed(0)
    points, sdf = to_torch(primitive_scenes(8, seed=0), 500)
    model = LatentSDF(n_ctrl=5, degree=1, code_length=2, decoder=mlp(4, 32, 2, "silu"))
    losses = []
    fit(
        model,
        points,
        sdf,
        steps=300,
        shapes_per_step=4,
        points_per_step=128,
        lr_codes=5e-3,
        lr_decoder=1e-3,
        delta=0.1,
        beta=1e-4,
        init_std=0.01,
        on_step=lambda step, loss: losses.append(loss.item()),
    )
    assert sum(losses[-20:]) < 0.5 * sum(losses[:20])
