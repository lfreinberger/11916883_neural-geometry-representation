"""Quantum decoder: data re-uploading circuit, simulated with pennylane."""

import pennylane as qml
import torch
from torch import nn


class QNN(nn.Module):
    def __init__(self, in_features: int, n_qubits: int = 5, blocks: int = 6):
        super().__init__()
        self.n_qubits = n_qubits
        self.blocks = blocks
        # qubit q gets input q % in_features, with more qubits than inputs some are used twice
        self.inputs = [q % in_features for q in range(n_qubits)]
        self.scale = nn.Parameter(torch.ones(blocks, n_qubits))
        self.angles = nn.Parameter(2 * torch.pi * torch.rand(blocks, n_qubits, 3))
        # <Z> is in [-1, 1], a = 0.1 maps it into [-0.1, 0.1] = inside the clamp of the loss.
        # outside the clamp there is no gradient (same reason as the zero init in mlp)
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))
        device = qml.device("default.qubit", wires=n_qubits)
        self.circuit = qml.QNode(self.circuit, device, interface="torch", diff_method="backprop")

    def circuit(self, x: torch.Tensor, scale: torch.Tensor, angles: torch.Tensor):
        # x (N, in_features)
        wires = range(self.n_qubits)
        start = torch.zeros(2**self.n_qubits, dtype=torch.complex64, device=x.device)
        start[0] = 1
        qml.StatePrep(start, wires=wires)
        for block in range(self.blocks):
            qml.AngleEmbedding(scale[block] * x[:, self.inputs], wires=wires, rotation="Y")
            # one layer = Rot on every qubit + CNOT(q, q + 1), CNOT(n - 1, 0) closes the ring
            qml.StronglyEntanglingLayers(angles[block : block + 1], wires=wires)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.circuit(x.reshape(-1, x.shape[-1]), self.scale, self.angles)
        return (self.a * z.to(x.dtype) + self.b).reshape(*x.shape[:-1], 1)
