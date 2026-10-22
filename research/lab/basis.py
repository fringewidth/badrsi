"""A random rotation of a layer's neuron space, as a control for whether the neuron basis is special.

A dense Haar rotation of 9,728 dimensions per layer would be 380 MB a layer. Instead each layer gets
R = P2 (A2 kron B2) P1 (A1 kron B1): two rounds of a Kronecker product of two random orthogonal matrices whose sizes
multiply to the layer width, with random permutations between, so every rotated unit is a dense mix of all neurons.
Applying it is two small matmuls on a reshaped vector.
"""

import mlx.core as mx
import numpy as np


def factor(n: int) -> tuple[int, int]:
    """The two factors of n closest to its square root."""
    a = int(np.sqrt(n))
    while n % a:
        a -= 1
    return a, n // a


def haar(n: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


class Rotation:
    def __init__(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        self.n = n
        self.a, self.b = factor(n)
        self.rounds = []
        for _ in range(2):
            A = mx.array(haar(self.a, rng).astype(np.float32))
            B = mx.array(haar(self.b, rng).astype(np.float32))
            perm = rng.permutation(n)
            self.rounds.append((A, B, mx.array(perm), mx.array(np.argsort(perm))))

    def _kron(self, x: mx.array, A: mx.array, B: mx.array, transpose: bool) -> mx.array:
        # (A kron B) applied to the last axis of x, viewed as [a, b]: A X B^T. Transposed: A^T X B.
        s = x.shape
        X = x.reshape(*s[:-1], self.a, self.b)
        Y = (A.T @ X @ B) if transpose else (A @ X @ B.T)
        return Y.reshape(s)

    def to_units(self, x: mx.array) -> mx.array:
        """Coordinates of neuron-space vectors x (last axis) in the rotated units: R^T x."""
        x = x.astype(mx.float32)
        # R = P2 K2 P1 K1, so R^T = K1^T P1^T K2^T P2^T. P as a matrix maps e_i to e_perm[i]; P^T x = x[perm].
        (A1, B1, p1, _), (A2, B2, p2, _) = self.rounds
        x = mx.take(x, p2, axis=-1)
        x = self._kron(x, A2, B2, True)
        x = mx.take(x, p1, axis=-1)
        return self._kron(x, A1, B1, True)

    def to_neurons(self, u: mx.array) -> mx.array:
        """Neuron-space vectors from rotated-unit coordinates u (last axis): R u."""
        (A1, B1, _, i1), (A2, B2, _, i2) = self.rounds
        x = self._kron(u.astype(mx.float32), A1, B1, False)
        x = mx.take(x, i1, axis=-1)
        x = self._kron(x, A2, B2, False)
        return mx.take(x, i2, axis=-1)
