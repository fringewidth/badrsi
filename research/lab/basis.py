"""A random rotation of a layer's neuron space, as a control for whether the neuron basis is special.

A dense Haar rotation of 9,728 dimensions per layer would be 380 MB a layer. Instead each layer gets
R = P2 (A2 kron B2) P1 (A1 kron B1): two rounds of a Kronecker product of two random orthogonal matrices whose sizes
multiply to the layer width, with random permutations between, so every rotated unit is a dense mix of all neurons.
Applying it is two small matmuls on a reshaped vector.

A few neurons per layer carry massive activations, a thousand times the median (Qwen3-4B layers 6 and 35). A full
rotation smears them into every unit of the layer, so every rotated unit there carries a sliver of them, and patching
or scoring those units is dominated by that. FixedRotation leaves the outlier neurons as they are and rotates the rest,
so the neuron-versus-rotation comparison is about the bulk of the neurons, not about a handful of outliers.
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


def free_size(n: int, f: int) -> int:
    """At least f fixed coordinates, padded so the rotated rest factors into two sides of 32 or more."""
    while factor(n - f)[0] < 32:
        f += 1
    return f


class FixedRotation:
    """Identity on the fixed neurons, a Rotation on the rest. Unit i is neuron i for fixed i."""

    def __init__(self, n: int, seed: int, fixed: list[int]):
        self.n = n
        fixed = sorted(set(fixed))
        free = [i for i in range(n) if i not in set(fixed)]
        self.m = len(free)
        order = np.array(free + fixed)
        self.order, self.inv = mx.array(order), mx.array(np.argsort(order))
        self.inner = Rotation(self.m, seed)
        self.fixed = fixed

    def to_units(self, x: mx.array) -> mx.array:
        z = mx.take(x.astype(mx.float32), self.order, axis=-1)
        z = mx.concatenate([self.inner.to_units(z[..., : self.m]), z[..., self.m :]], axis=-1)
        return mx.take(z, self.inv, axis=-1)

    def to_neurons(self, u: mx.array) -> mx.array:
        z = mx.take(u.astype(mx.float32), self.order, axis=-1)
        z = mx.concatenate([self.inner.to_neurons(z[..., : self.m]), z[..., self.m :]], axis=-1)
        return mx.take(z, self.inv, axis=-1)
