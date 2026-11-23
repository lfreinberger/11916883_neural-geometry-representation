"""Train the decoder on primitive scenes, then reconstruct held-out scenes and MPEG-7 val.

uv run python -m neural_geometry.train                 # baseline V0
uv run python -m neural_geometry.train model=spline    # B-spline reference, nothing to train
"""

import logging
from pathlib import Path

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from shapely.geometry.base import BaseGeometry

from neural_geometry import mpeg7
from neural_geometry.fit import clamped_l1, fit
from neural_geometry.model import LatentSDF, mlp
from neural_geometry.shapes import primitive_scenes, sample_dataset

log = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> LatentSDF:
    decoder = None
    if cfg.decoder == "mlp":
        decoder = mlp(cfg.code_length + 2, cfg.hidden, cfg.layers, cfg.activation)
    return LatentSDF(cfg.n_ctrl, cfg.degree, cfg.code_length, decoder)


def to_tensors(
    shapes: list[BaseGeometry], n_points: int, seed: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    points, sdf = sample_dataset(shapes, n_points, seed)
    return torch.from_numpy(points).to(device), torch.from_numpy(sdf).to(device)


def flat_params(cfg: DictConfig) -> dict:
    params = {}
    for key, value in OmegaConf.to_container(cfg, resolve=True).items():
        if isinstance(value, dict):
            params.update({f"{key}.{k}": v for k, v in value.items()})
        else:
            params[key] = value
    return params


def log_every(name: str, total: int, every: int = 100):
    def on_step(step: int, loss: torch.Tensor) -> None:
        if step % every == 0:
            mlflow.log_metric(f"{name}_loss", loss.item(), step=step)
        if step % (10 * every) == 0 or step == total - 1:
            log.info(f"{name} {step + 1}/{total}  loss {loss.item():.2e}")

    return on_step


def reconstruct(
    model: LatentSDF,
    points: torch.Tensor,
    sdf: torch.Tensor,
    cfg: DictConfig,
    on_step=None,
) -> tuple[torch.Tensor, float]:
    codes = fit(
        model,
        points,
        sdf,
        **cfg.reconstruct,
        **cfg.loss,
        seed=cfg.seed,
        on_step=on_step,
    )
    with torch.no_grad():
        error = clamped_l1(model(codes, points), sdf, cfg.loss.delta)
    return codes, error.item()


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    model = build_model(cfg.model).to(device)
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    mpeg7_dir = Path(cfg.data.mpeg7_dir)
    val_names = mpeg7.split(mpeg7.shape_names(mpeg7_dir))["val"]
    targets = {
        # held-out scenes come from the training distribution (overfitting check)
        "heldout": primitive_scenes(cfg.data.n_heldout, cfg.data.heldout_seed),
        "mpeg7_val": [mpeg7.load_shape(mpeg7_dir / name) for name in val_names],
    }
    val_data = {
        name: to_tensors(shapes, cfg.data.n_points_val, cfg.data.heldout_seed, device)
        for name, shapes in targets.items()
    }

    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=cfg.model.name):
        mlflow.log_params(flat_params(cfg))
        mlflow.log_param("budget", cfg.model.n_ctrl**2 * cfg.model.code_length)

        steps = 0
        if model.decoder is not None:
            steps = cfg.train.steps
            log_train = log_every("train", steps)

            def on_train_step(step: int, loss: torch.Tensor) -> None:
                log_train(step, loss)
                # every validation is a full reconstruction, so not too often
                if cfg.eval_every and (step + 1) % cfg.eval_every == 0 and step + 1 < steps:
                    for name, (points, sdf) in val_data.items():
                        _, error = reconstruct(model, points, sdf, cfg)
                        mlflow.log_metric(f"{name}_l1", error, step=step + 1)
                        log.info(f"{name} l1 {error:.2e}")

            scenes = primitive_scenes(cfg.data.n_train, cfg.data.train_seed)
            points, sdf = to_tensors(scenes, cfg.data.n_points, cfg.data.train_seed, device)
            train_codes = fit(
                model,
                points,
                sdf,
                **cfg.train,
                **cfg.loss,
                seed=cfg.seed,
                on_step=on_train_step,
            )
            torch.save(model.state_dict(), out_dir / "model.pt")
            torch.save(train_codes.cpu(), out_dir / "codes_train.pt")
            mlflow.log_artifact(str(out_dir / "model.pt"))
            mlflow.log_artifact(str(out_dir / "codes_train.pt"))

        for name, (points, sdf) in val_data.items():
            log.info(f"reconstructing {name} ({len(points)} shapes)")
            codes, error = reconstruct(
                model, points, sdf, cfg, on_step=log_every(name, cfg.reconstruct.steps)
            )
            mlflow.log_metric(f"{name}_l1", error, step=steps)
            log.info(f"{name} l1 {error:.2e}")
            torch.save(codes.cpu(), out_dir / f"codes_{name}.pt")
            mlflow.log_artifact(str(out_dir / f"codes_{name}.pt"))


if __name__ == "__main__":
    main()
