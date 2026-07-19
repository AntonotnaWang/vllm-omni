# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Triton kernel constexpr params use uppercase (BLOCK_M, BLOCK_D, D) by convention.
# ruff: noqa: N803
#
# Fused online-softmax merge for Ring Attention (inference / forward only).
#
# This is *new* optimization code — it does NOT replace ``ring_utils.py``.
# ``ring_utils.update_out_and_lse`` (the original, defensive torch path) is
# still used by the baseline ``ring_flash_attn_forward_org``.  This module provides
# a single fused Triton kernel that collapses the per-step online-softmax merge
# (baseline: several elementwise kernels for sigmoid/logsigmoid/sub/mul + a
# bf16<->fp32 round-trip) into one launch that writes in place on fp32
# accumulators, and can write the *final* step straight to the bf16 output.
#
# Numerics are identical to ``ring_utils._update_out_and_lse``:
#     out = out * (1 - sig) + block_out * sig       , sig = sigmoid(block_lse - lse)
#     lse = lse + softplus(block_lse - lse)          (== lse - logsigmoid(lse - block_lse))
#
# Adapted from the sigmoid/softplus merge in
#   ring-flash-attention/ring_flash_attn/triton_utils.py
# combined with the is_first / is_final direct-bf16-write trick from
#   large_model_from_scratch/optimized_ring_flash_attn_v2.py

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON_MERGE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - env without triton
    triton = None
    tl = None
    HAS_TRITON_MERGE = False


if HAS_TRITON_MERGE:

    @triton.jit
    def _ring_merge_kernel(
        block_out_ptr,  # (B, S, H, D)   compute dtype, contiguous
        block_lse_ptr,  # (B*S*H,)       fp32, row-aligned to out (see wrapper)
        out_acc_ptr,  # (B, S, H, D)   fp32 accumulator, contiguous
        lse_acc_ptr,  # (B, S, H, 1)   fp32 accumulator, contiguous
        out_ptr,  # (B, S, H, D)   output dtype (e.g. bf16), contiguous
        n_rows,  # B * S * H
        is_first: tl.constexpr,  # first valid step: nothing to merge with
        is_final: tl.constexpr,  # last valid step: write result to out_ptr (bf16)
        D: tl.constexpr,  # head_dim
        BLOCK_M: tl.constexpr,  # rows per program
        BLOCK_D: tl.constexpr,  # next_pow2(head_dim)
    ):
        pid = tl.program_id(axis=0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        rmask = rows < n_rows
        mask = rmask[:, None] & (cols[None, :] < D)

        bo = tl.load(block_out_ptr + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        blse = tl.load(block_lse_ptr + rows, mask=rmask, other=0.0)

        if is_first:
            merged_out = bo
            merged_lse = blse
        else:
            old_out = tl.load(out_acc_ptr + rows[:, None] * D + cols[None, :], mask=mask, other=0.0)
            old_lse = tl.load(lse_acc_ptr + rows, mask=rmask, other=0.0)
            s = blse - old_lse
            sig = tl.sigmoid(s)
            merged_out = old_out * (1.0 - sig[:, None]) + bo * sig[:, None]
            # softplus(s) = max(s, 0) + log1p(exp(-|s|))  (stable)
            merged_lse = old_lse + tl.maximum(s, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(s)))

        # lse accumulator is always kept in fp32.
        tl.store(lse_acc_ptr + rows, merged_lse, mask=rmask)

        if is_final:
            tl.store(
                out_ptr + rows[:, None] * D + cols[None, :],
                merged_out.to(out_ptr.dtype.element_ty),
                mask=mask,
            )
        else:
            tl.store(out_acc_ptr + rows[:, None] * D + cols[None, :], merged_out, mask=mask)

    def fused_merge(
        out_acc: torch.Tensor,  # (B, S, H, D) fp32, in/out accumulator
        lse_acc: torch.Tensor,  # (B, S, H, 1) fp32, in/out accumulator
        out: torch.Tensor,  # (B, S, H, D) output dtype, written on final step
        block_out: torch.Tensor,  # (B, S, H, D) compute dtype
        block_lse: torch.Tensor,  # (B, H, S) fp32
        is_first: bool,
        is_final: bool,
    ) -> None:
        """Fused online-softmax merge of one ring block into the accumulators.

        In place on ``out_acc`` / ``lse_acc``.  When ``is_final`` the merged
        result is written directly to ``out`` (its dtype, e.g. bf16), so the
        caller avoids a trailing full-tensor ``out_acc.to(dtype)`` copy.
        """
        B, S, H, D = out_acc.shape
        n_rows = B * S * H
        # Reorder block_lse (B, H, S) -> row order of out (B, S, H) and flatten.
        # Cheap: n_rows fp32 elements. Keeps the kernel a simple 1-D indexer.
        blse_flat = block_lse.transpose(1, 2).reshape(-1).contiguous()
        if not block_out.is_contiguous():
            block_out = block_out.contiguous()

        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_M = max(1, 8192 // BLOCK_D)
        grid = (triton.cdiv(n_rows, BLOCK_M),)
        _ring_merge_kernel[grid](
            block_out,
            blse_flat,
            out_acc,
            lse_acc,
            out,
            n_rows,
            is_first,
            is_final,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )

else:  # pragma: no cover - env without triton

    def fused_merge(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Triton is not available; fused_merge cannot be used.")
