"""
Negacyclic Number Theoretic Transform (NTT) implementation.

The negacyclic NTT computes polynomial evaluation at odd powers of a primitive
root. Given coefficients x[0], x[1], ..., x[N-1], the output is:

    y[k] = Σ_{n=0}^{N-1} x[n] · ψ^{(2k+1)·n}  (mod q)

where ψ is a primitive 2N-th root of unity (ψ^N ≡ -1 mod q).

This is equivalent to a cyclic NTT on "twisted" input, where each coefficient
x[n] is first multiplied by ψ^n.
"""

import jax.numpy as jnp



# -----------------------------------------------------------------------------
# Modular Arithmetic
# -----------------------------------------------------------------------------

def mod_add(a, b, q):
    """Return (a + b) mod q, elementwise."""
    res = a + b
    return jnp.where(res >= q, res - q, res)

def mod_sub(a, b, q):
    """Return (a - b) mod q, elementwise."""
    return jnp.where(a < b, a + q - b, a - b)


def mod_mul(a, b, q):
    """Return (a * b) mod q, elementwise."""
    return jnp.mod(a.astype(jnp.uint64) * b.astype(jnp.uint64), q.astype(jnp.uint64)).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------
def _montgomery_reduce(T, q, q_inv):
    """Reduces a 64-bit product T back to a 32-bit Montgomery representation."""
    m = jnp.uint32(T) * q_inv
    t = (T.astype(jnp.uint64) + m.astype(jnp.uint64) * q.astype(jnp.uint64)) >> 32
    t = t.astype(jnp.uint32)
    return jnp.where(t >= q, t - q, t)

def _montgomery_mul(a, b, q, q_inv):
    """Multiplies two numbers in Montgomery form."""
    T = a.astype(jnp.uint64) * b.astype(jnp.uint64)
    return _montgomery_reduce(T, q, q_inv)

def ntt(x, *, q, psi_powers, twiddles):
    """
    Compute the forward negacyclic NTT.

    Args:
        x: Input coefficients, shape (batch, N), values in [0, q)
        q: Prime modulus satisfying (q - 1) % 2N == 0
        psi_powers: Precomputed ψ^n table
        twiddles: Precomputed twiddle table

    Returns:
        jnp.ndarray: NTT output, same shape as input
    """
    N = x.shape[-1]
    x = x.astype(jnp.uint32)
    q_u32 = jnp.uint32(q)

    # 1. Compute q_inv for Montgomery reduction (q * q_inv = -1 mod 2^32)
    # Using 5 Newton-Raphson iterations is enough for 32-bit integers
    q_inv = jnp.uint32(1)
    for _ in range(5):
        q_inv = q_inv * (jnp.uint32(2) - q_u32 * q_inv)
    q_inv = -q_inv

    # 2. Negacyclic twist AND automatic conversion into Montgomery domain
    # Since psi_powers was scaled by R^2 in prepare_tables, the result is x * psi * R (mod q)
    x = _montgomery_mul(x, psi_powers, q_u32, q_inv)

    # 3. Bit-reverse the input for Cooley-Tukey DIT
    bits = N.bit_length() - 1
    # JAX trick: Reshape to binary bits, transpose the bits, and flatten
    x_reshaped = x.reshape((x.shape[0],) + (2,) * bits)
    axes = (0,) + tuple(range(bits, 0, -1))
    x = jnp.transpose(x_reshaped, axes=axes)
    x = x.reshape(x.shape[0], N)

    # 4. Cooley-Tukey DIT NTT stages
    for s in range(bits):
        span = 1 << s
        W = twiddles[span : 2*span].reshape(1, 1, span)

        x_resh = x.reshape(x.shape[0], N // (2 * span), 2, span)
        u = x_resh[:, :, 0, :]
        v = x_resh[:, :, 1, :]

        # Multiply by twiddle factor (keeps numbers in Montgomery form)
        v_twiddled = _montgomery_mul(v, W, q_u32, q_inv)

        # Butterfly add/sub
        even = mod_add(u, v_twiddled, q_u32)
        odd  = mod_sub(u, v_twiddled, q_u32)

        # Recombine
        x = jnp.stack([even, odd], axis=2).reshape(x.shape[0], N)

    # 5. Convert out of Montgomery form back to normal integer domain
    x = _montgomery_reduce(x, q_u32, q_inv)

    return x


def prepare_tables(*, q, psi_powers, twiddles):
    """
    Optional one-time table preparation.

    Override this if you want faster modular multiplication than JAX's "%".
    For example, you can convert the provided tables into Montgomery form
    (or any other domain) once here, then run `ntt` using your mod_mul.
    This function runs before timing, so its cost is not counted as latency.
    Must return (psi_powers, twiddles) in the form expected by `ntt`.
    """
    q_int = int(q)
    
    # Compute R = 2^32 mod q, and R^2 mod q
    R = (1 << 32) % q_int
    R2 = (R * R) % q_int

    # Scale psi_powers by R^2 so the initial twist also converts x to Montgomery form
    psi_powers_opt = jnp.mod(psi_powers.astype(jnp.uint64) * R2, q).astype(jnp.uint32)

    # Convert twiddles directly into standard Montgomery form (* R)
    twiddles_mont = jnp.mod(twiddles.astype(jnp.uint64) * R, q).astype(jnp.uint32)

    return psi_powers_opt, twiddles_mont

