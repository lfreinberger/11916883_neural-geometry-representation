"""Auto-decoder training and reconstruction with the clamped L1 loss of DeepSDF."""

from collections.abc import Callable

import torch

from neural_geometry.model import LatentSDF


def clamped_l1(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    return (pred.clamp(-delta, delta) - target.clamp(-delta, delta)).abs().mean()


def fit(
    model: LatentSDF,
    points: torch.Tensor,
    sdf: torch.Tensor,
    *,
    steps: int,
    lr_codes: float,
    points_per_step: int,
    delta: float,
    beta: float,
    lr_decoder: float = 0.0,
    shapes_per_step: int | None = None,
    init_std: float = 0.0,
    unclamped_warmup: float = 0.0,
    lr_decay_at: float = 1.0,
    lr_decay: float = 1.0,
    seed: int = 0,
    on_step: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Fit one code grid per shape to points (S, N, 2) and sdf (S, N).

    With lr_decoder > 0 the decoder is trained as well (training), otherwise it is
    frozen (reconstruction). Returns the codes (S, n_ctrl, n_ctrl, code_length).
    """
    device = points.device
    generator = torch.Generator(device).manual_seed(seed)
    n_shapes, n_points = sdf.shape
    shape = (n_shapes, model.n_ctrl, model.n_ctrl, model.code_length)
    codes = (init_std * torch.randn(shape, generator=generator, device=device)).requires_grad_()

    model.requires_grad_(lr_decoder > 0)
    groups = [{"params": [codes], "lr": lr_codes}]
    if lr_decoder > 0:
        groups.append({"params": model.parameters(), "lr": lr_decoder})
    optimizer = torch.optim.Adam(groups)
    # one step decay, lr * lr_decay after lr_decay_at of the steps (default off)
    milestone = int(lr_decay_at * steps)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [milestone], gamma=lr_decay)

    for step in range(steps):
        batch = torch.randperm(n_shapes, generator=generator, device=device)[:shapes_per_step]
        idx = torch.randint(
            n_points, (len(batch), points_per_step), generator=generator, device=device
        )
        rows = batch[:, None]
        pred = model(codes[batch], points[rows, idx])
        # sum over the control points as in the paper, mean over shapes
        reg = codes[batch].pow(2).sum(dim=(1, 2, 3)).mean()
        # no clamp at the start: if the decoder starts outside +-delta with the wrong sign,
        # the clamped loss has no gradient there and the reconstruction never recovers
        clamp = delta if step >= unclamped_warmup * steps else float("inf")
        loss = clamped_l1(pred, sdf[rows, idx], clamp) + beta * reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if on_step is not None:
            on_step(step, loss.detach())
    # fit can run inside another fit (validation during training)
    model.requires_grad_(True)
    return codes.detach()
