"""
Assignment 2 student implementation reference skeleton.

This file documents the frozen student-facing API.
Only 32-bit kernels are compulsory in the base track.
64-bit and 128-bit kernels are intentionally left unimplemented here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import functools

jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# 32-bit primitives (compulsory)
# -----------------------------------------------------------------------------

def mod_add_32(a, b, q):
    """Return (a + b) mod q for the 32-bit track."""
    a_64 = jnp.asarray(a, dtype=jnp.uint64)
    b_64 = jnp.asarray(b, dtype=jnp.uint64)
    q_64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.mod(a_64 + b_64, q_64).astype(jnp.uint32)


def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    # Subtraction must remain signed int64 to correctly handle negative intermediate values
    a_64 = jnp.asarray(a, dtype=jnp.int64)
    b_64 = jnp.asarray(b, dtype=jnp.int64)
    q_64 = jnp.asarray(q, dtype=jnp.int64)
    return jnp.mod(a_64 - b_64, q_64).astype(jnp.uint32)


def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    # Multiplication must be uint64 to avoid overflow from max 32-bit products
    a_64 = jnp.asarray(a, dtype=jnp.uint64)
    b_64 = jnp.asarray(b, dtype=jnp.uint64)
    q_64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.mod(a_64 * b_64, q_64).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_64(a, b, q):
    """Optional 64-bit modular add kernel."""
    raise NotImplementedError


def mod_sub_64(a, b, q):
    """Optional 64-bit modular subtract kernel."""
    raise NotImplementedError


def mod_mul_64(a, b, q):
    """Optional 64-bit modular multiply kernel."""
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 128-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_128(a, b, q):
    """Optional 128-bit modular add kernel."""
    raise NotImplementedError


def mod_sub_128(a, b, q):
    """Optional 128-bit modular subtract kernel."""
    raise NotImplementedError


def mod_mul_128(a, b, q):
    """Optional 128-bit modular multiply kernel."""
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Frozen dispatch API
# -----------------------------------------------------------------------------

def mod_add(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_add_32(a, b, q)
    if int(bit_width) == 64:
        return mod_add_64(a, b, q)
    if int(bit_width) == 128:
        return mod_add_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_sub(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_sub_32(a, b, q)
    if int(bit_width) == 64:
        return mod_sub_64(a, b, q)
    if int(bit_width) == 128:
        return mod_sub_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_mul(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_mul_32(a, b, q)
    if int(bit_width) == 64:
        return mod_mul_64(a, b, q)
    if int(bit_width) == 128:
        return mod_mul_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """Compulsory 32-bit MLE update."""
    diff = mod_sub_32(one_eval, zero_eval, q)
    prod = mod_mul_32(diff, target_eval, q)
    return mod_add_32(zero_eval, prod, q)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    """Optional 64-bit MLE update."""
    raise NotImplementedError


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    """Optional 128-bit MLE update."""
    raise NotImplementedError


def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# -----------------------------------------------------------------------------
# JIT Compiled Core Logic
# -----------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=["expression", "num_rounds"])
def _sumcheck_32_jit(eval_tables, q, expression, challenges, num_rounds):
    # Calculate degree of the composition
    degree = max((len(term) for term in expression), default=1)

    # Initialize tables and variables into strictly uint32 types
    current_tables = {k: jnp.asarray(v, dtype=jnp.uint32) for k, v in eval_tables.items()}
    challenges_arr = jnp.asarray(challenges, dtype=jnp.uint32)
    q_arr = jnp.asarray(q, dtype=jnp.uint32)

    round_evals = []

    for i in range(num_rounds):
        # We reshape array dynamically instead of striding. Shape translates to (-1, 2) implicitly 
        table_reshaped = {k: jnp.reshape(v, (-1, 2)) for k, v in current_tables.items()}
        table_0 = {k: v[:, 0] for k, v in table_reshaped.items()}
        table_1 = {k: v[:, 1] for k, v in table_reshaped.items()}

        # Optimization: Pre-compute the linear step size (o - z) for the entire block
        if degree >= 1:
            step = {var: mod_sub_32(table_1[var], table_0[var], q_arr) for var in current_tables.keys()}
        else:
            step = {}

        evals_i = []
        t_k = table_0
        
        # Calculate g_i(k) for k in [0..degree] using optimized sequential additions
        for k in range(degree + 1):
            if k == 0:
                t_k = table_0
            elif k == 1:
                t_k = table_1
            else:
                t_k = {var: mod_add_32(t_k[var], step[var], q_arr) for var in current_tables.keys()}

            # Evaluate composite polynomial element-wise
            sum_evals = jnp.zeros_like(next(iter(t_k.values())), dtype=jnp.uint32)
            for term in expression:
                term_prod = jnp.ones_like(sum_evals, dtype=jnp.uint32)
                for var in term:
                    term_prod = mod_mul_32(term_prod, t_k[var], q_arr)
                sum_evals = mod_add_32(sum_evals, term_prod, q_arr)

            # Cast locally to uint64 strictly to bypass sum overflow of the resulting elements
            sum_evals_64 = jnp.sum(jnp.asarray(sum_evals, dtype=jnp.uint64))
            k_eval = jnp.mod(sum_evals_64, jnp.uint64(q)).astype(jnp.uint32)
            evals_i.append(k_eval)

        round_evals.append(jnp.stack(evals_i))

        # Perform table update using random challenge generated at current index
        if i < num_rounds - 1:
            r_i = challenges_arr[i]
            current_tables = {
                var: mle_update_32(table_0[var], table_1[var], r_i, q=q_arr)
                for var in current_tables.keys()
            }

    round_evals_stack = jnp.stack(round_evals)
    claim0 = mod_add_32(round_evals_stack[0, 0], round_evals_stack[0, 1], q_arr)

    return claim0, round_evals_stack


def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """Compulsory 32-bit sumcheck path."""
    # Convert inner expression lists to tuples to allow them to be passed as statically hashed JIT args
    hashable_expr = tuple(tuple(term) for term in expression)
    return _sumcheck_32_jit(eval_tables, q, hashable_expr, challenges, num_rounds)


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 64-bit sumcheck path."""
    raise NotImplementedError


def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 128-bit sumcheck path."""
    raise NotImplementedError


def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    """Frozen dispatcher entrypoint used by the harness."""
    if int(bit_width) == 32:
        return sumcheck_32(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 64:
        return sumcheck_64(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 128:
        return sumcheck_128(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    raise ValueError(f"Unsupported bit_width={bit_width}")