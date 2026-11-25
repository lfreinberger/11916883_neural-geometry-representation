import torch

from neural_geometry.fit import fit
from neural_geometry.model import LatentSDF
from neural_geometry.quantum import QNN
from neural_geometry.shapes import primitive_scenes, sample_dataset


def test_qnn_size_and_start():
    torch.manual_seed(0)
    qnn = QNN(in_features=4, n_qubits=5, blocks=6)
    assert sum(p.numel() for p in qnn.parameters()) == 122
    out = qnn(2 * torch.rand(3, 100, 4) - 1)
    assert out.shape == (3, 100, 1)
    # inside the clamp of the loss, otherwise no gradient (dead start)
    assert out.abs().max() <= 0.1 + 1e-6


def test_tiny_qnn_training_run_decreases_the_loss():
    torch.manual_seed(0)
    points, sdf = sample_dataset(primitive_scenes(8, seed=0), 500, seed=0)
    model = LatentSDF(n_ctrl=5, degree=1, code_length=2, decoder=QNN(4, n_qubits=5, blocks=3))
    losses = []
    fit(
        model,
        torch.from_numpy(points),
        torch.from_numpy(sdf),
        steps=300,
        shapes_per_step=4,
        points_per_step=128,
        lr_codes=5e-3,
        lr_decoder=5e-3,
        delta=0.1,
        beta=0.0,
        init_std=0.01,
        on_step=lambda step, loss: losses.append(loss.item()),
    )
    assert sum(losses[-20:]) < 0.5 * sum(losses[:20])
