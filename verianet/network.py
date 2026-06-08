"""Network weight loading and forward evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .activations import gelu
from .paths import WEIGHTS_PATH

Array = np.ndarray


@dataclass(frozen=True)
class NetworkWeights:
    """Weights for the 49 -> 3 -> 3 -> 10 Verianet MLP."""

    W1: Array
    b1: Array
    W2: Array
    b2: Array
    W3: Array
    b3: Array

    @classmethod
    def load(cls, path: str | Path | None = None) -> "NetworkWeights":
        data = np.load(Path(path) if path is not None else WEIGHTS_PATH)
        weights = cls(
            W1=np.asarray(data["W1"], dtype=np.float64),
            b1=np.asarray(data["b1"], dtype=np.float64),
            W2=np.asarray(data["W2"], dtype=np.float64),
            b2=np.asarray(data["b2"], dtype=np.float64),
            W3=np.asarray(data["W3"], dtype=np.float64),
            b3=np.asarray(data["b3"], dtype=np.float64),
        )
        weights.validate()
        return weights

    @property
    def input_dim(self) -> int:
        return int(self.W1.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.W3.shape[1])

    def validate(self) -> None:
        checks = [
            (self.W1.ndim == 2, "W1 must be rank 2"),
            (self.W2.ndim == 2, "W2 must be rank 2"),
            (self.W3.ndim == 2, "W3 must be rank 2"),
            (self.b1.shape == (self.W1.shape[1],), "b1 shape must match W1 output"),
            (self.W2.shape[0] == self.W1.shape[1], "W2 input must match W1 output"),
            (self.b2.shape == (self.W2.shape[1],), "b2 shape must match W2 output"),
            (self.W3.shape[0] == self.W2.shape[1], "W3 input must match W2 output"),
            (self.b3.shape == (self.W3.shape[1],), "b3 shape must match W3 output"),
        ]
        errors = [msg for ok, msg in checks if not ok]
        if errors:
            raise ValueError("; ".join(errors))

    def forward_logits(self, x: Array) -> Array:
        x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_flat.size != self.input_dim:
            raise ValueError(f"expected {self.input_dim} input values, got {x_flat.size}")
        a1 = x_flat @ self.W1 + self.b1
        z1 = gelu(a1)
        a2 = z1 @ self.W2 + self.b2
        z2 = gelu(a2)
        return z2 @ self.W3 + self.b3

    def predict(self, x: Array) -> int:
        return int(np.argmax(self.forward_logits(x)))
