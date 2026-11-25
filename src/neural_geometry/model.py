"""Latent field SDF shared by all models, its mlp decoder and the B-spline reference."""

import torch
from torch import nn


def bspline_basis(x: torch.Tensor, n_ctrl: int, degree: int) -> torch.Tensor:
    """Clamped uniform B-spline basis on [-1, 1], shape (...) -> (..., n_ctrl)."""
    inner = torch.linspace(-1, 1, n_ctrl - degree + 1, dtype=x.dtype, device=x.device)
    knots = torch.cat([inner[:1].repeat(degree), inner, inner[-1:].repeat(degree)])
    # spans are half-open, so x = 1 would get no basis function
    x = x.clamp(-1, 1 - 1e-6)[..., None]
    basis = ((knots[:-1] <= x) & (x < knots[1:])).to(x.dtype)
    # cox-de boor
    for k in range(1, degree + 1):
        left = _ratio(x - knots[: -k - 1], knots[k:-1] - knots[: -k - 1])
        right = _ratio(knots[k + 1 :] - x, knots[k + 1 :] - knots[1:-k])
        basis = left * basis[..., :-1] + right * basis[..., 1:]
    return basis


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    # 0/0 = 0 at the repeated end knots
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-12), 0.0)


def hat(x: torch.Tensor, n_tiles: int) -> torch.Tensor:
    """Local coordinate in [-1, 1] per tile, mirrored in every second tile."""
    s = (x + 1) / 2 * n_tiles
    return 1 - 2 * torch.abs(torch.remainder(s, 2) - 1)


def mlp(in_features: int, hidden: int, layers: int, activation: str = "relu") -> nn.Sequential:
    act = {"relu": nn.ReLU, "silu": nn.SiLU}[activation]
    modules = []
    for i in range(layers):
        modules += [nn.Linear(in_features if i == 0 else hidden, hidden), act()]
    out = nn.Linear(hidden, 1)
    # start at sdf = 0, inside the clamp of the loss. A decoder that starts outside
    # +-delta everywhere gets no gradient at all (with relu)
    nn.init.zeros_(out.weight)
    nn.init.zeros_(out.bias)
    return nn.Sequential(*modules, out)


class LatentSDF(nn.Module):
    """sdf(x) = decoder(lambda(x), hat(x)) with a B-spline latent field lambda.

    Without decoder the field itself is the sdf (code_length 1), i.e. a classical
    B-spline SDF.
    """

    def __init__(
        self, n_ctrl: int, degree: int, code_length: int, decoder: nn.Module | None = None
    ):
        super().__init__()
        self.n_ctrl = n_ctrl
        self.degree = degree
        self.code_length = code_length
        self.n_tiles = n_ctrl - degree
        self.decoder = decoder

    def forward(self, codes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """codes (B, n_ctrl, n_ctrl, code_length), points (B, P, 2) -> sdf (B, P)."""
        bx = bspline_basis(points[..., 0], self.n_ctrl, self.degree)
        by = bspline_basis(points[..., 1], self.n_ctrl, self.degree)
        latent = torch.einsum("bpi,bpj,bijc->bpc", bx, by, codes)
        if self.decoder is None:
            return latent[..., 0]
        local = hat(points, self.n_tiles)
        return self.decoder(torch.cat([latent, local], dim=-1))[..., 0]
