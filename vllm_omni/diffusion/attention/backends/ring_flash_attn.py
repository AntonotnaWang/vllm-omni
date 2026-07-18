# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention


import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType, select_flash_attn_impl
from vllm_omni.diffusion.attention.backends.ring.ring_triton import HAS_TRITON_MERGE, fused_merge
from vllm_omni.diffusion.attention.backends.ring.ring_utils import update_out_and_lse
from vllm_omni.diffusion.distributed.comm import RingComm


def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
):
    # Validate causal + joint_strategy combination
    # When causal=True and joint_strategy="rear", the causal mask would incorrectly
    # prevent local query tokens from attending to joint key tokens (which are
    # concatenated at the end). This breaks the semantics where joint tokens
    # (e.g., text conditioning) should be visible to all local tokens.
    if causal and joint_tensor_key is not None and joint_strategy == "rear":
        raise ValueError(
            "joint_strategy='rear' is not compatible with causal=True in Ring Attention. "
            "When using causal attention with joint tokens, use joint_strategy='front' "
            "to ensure joint tokens act as a visible prefix for all local tokens. "
            "With 'rear' strategy, the causal mask would incorrectly block local tokens "
            "from seeing the joint tokens."
        )

    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    # Check and adjust q, k, v to be contiguous
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor
            next_v: torch.Tensor
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        if not causal or step <= comm.rank:
            step_k = k
            step_v = v
            if step == 0 and joint_tensor_key is not None:
                if joint_strategy == "front":
                    step_k = torch.cat([joint_tensor_key, step_k], dim=1)
                    step_v = torch.cat([joint_tensor_value, step_v], dim=1)
                else:
                    step_k = torch.cat([step_k, joint_tensor_key], dim=1)
                    step_v = torch.cat([step_v, joint_tensor_value], dim=1)

            fn = select_flash_attn_impl(attn_type, stage="fwd-only", attn_processor=attn_processor)
            block_out, block_lse = fn(
                q,
                step_k,
                step_v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )

            # Ensure block_out is contiguous if needed, though usually it is from FA

            if attn_type == AttnType.SPARSE_SAGE:
                out, lse = block_out, block_lse
            else:
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    if attn_type != AttnType.SPARSE_SAGE:
        lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


class RingFlashAttnFunc(torch.autograd.Function):
    """Ring Flash Attention autograd function (inference only, no backward)."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_type,
        attn_processor,
        joint_tensor_key=None,
        joint_tensor_value=None,
        joint_strategy="front",
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out, softmax_lse = ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_type=attn_type,
            attn_processor=attn_processor,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        return out if not return_softmax else (out, softmax_lse, None)


def ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return RingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        None,  # attn_processor
        None,  # joint_tensor_key
        None,  # joint_tensor_value
        "front",  # joint_strategy
    )


def ring_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return RingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        None,  # attn_processor
        None,  # joint_tensor_key
        None,  # joint_tensor_value
        "front",  # joint_strategy
    )


def ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, None]:
    """Ring Attention forward pass using Flash Attention backend.

    Implements Ring Attention with sequence parallelism using a ring-based P2P
    communication pattern. The sequence dimension is sharded across devices, and
    Key/Value blocks are circulated through the ring to accumulate attention results.

    Args:
        q (torch.Tensor): Query tensor of shape (batch, seq_len, num_heads, head_dim).
            Sequence dimension is sharded across the ring group.
        k (torch.Tensor): Key tensor of shape (batch, seq_len, num_heads, head_dim).
            Sequence dimension is sharded across the ring group.
        v (torch.Tensor): Value tensor of shape (batch, seq_len, num_heads, head_dim).
            Sequence dimension is sharded across the ring group.
        dropout_p (float): Dropout probability. Defaults to 0.0.
        softmax_scale (float | None): Scaling factor for softmax.
            If None, computed as head_dim^(-0.5).
        causal (bool): Whether to apply causal masking. Defaults to False.
        window_size (tuple[int, int]): Sliding window size for attention.
            (-1, -1) means no windowing.
        softcap (float): Soft capping value for attention logits. Defaults to 0.0.
        alibi_slopes (torch.Tensor | None): ALiBi slopes for positional bias.
            Not supported.
        deterministic (bool): Whether to use deterministic algorithms.
            Defaults to False.
        return_attn_probs (bool): If True, returns (out, softmax_lse, None).
            Defaults to False.
        group (ProcessGroup | None): Process group for ring communication.
            Defaults to None.
        attn_type (AttnType): Flash Attention implementation type
            (AttnType.FA, AttnType.FA3, etc.).
        attn_processor (Callable | None): Custom attention processor for sparse
            attention. Defaults to None.
        joint_tensor_key (torch.Tensor | None): Additional key tensor for joint
            attention (e.g., text + image). Concatenated only at step=0.
            Defaults to None.
        joint_tensor_value (torch.Tensor | None): Additional value tensor for
            joint attention (e.g., text + image). Concatenated only at step=0.
            Defaults to None.
        joint_strategy (str): Concatenation strategy ("front" or "back").
            Defaults to "front".

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
            - If return_attn_probs is False: Output tensor (batch, seq_len, num_heads, head_dim).
            - If return_attn_probs is True: A tuple (out, softmax_lse, None).
    """
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        attn_processor,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    )


# ===========================================================================
# Optimized Ring Flash Attention (forward / inference only)
# ---------------------------------------------------------------------------
# The code below is ADDED alongside the baseline above (nothing removed) so the
# two can be benchmarked head-to-head. It layers three optimizations on top of
# the baseline path:
#   1. Packed-KV, double-buffered ring comm  -> 2 P2P ops per hop (was 4) and
#      zero per-hop allocation.
#   2. Fused Triton online-softmax merge     -> one kernel per step instead of
#      several elementwise kernels + bf16<->fp32 round trip.
#   3. Persistent fp32 workspace + direct bf16 write on the final step.
# Semantics (joint front/rear, causal step-0, SPARSE_SAGE bypass, GQA, return
# contract) are identical to the baseline; when Triton is unavailable or the
# backend has no mergeable LSE (SPARSE_SAGE) it transparently delegates to the
# baseline ``ring_flash_attn_forward``.
# ===========================================================================
class DoubleBufRingComm:
    """Double-buffered ring P2P that ships K and V packed into one tensor.

    Baseline :class:`RingComm` issues 4 P2P ops per hop (isend/irecv for K and V
    separately) and allocates a fresh recv buffer each hop. Packing K/V into a
    single ``[2, B, S, H_kv, D]`` tensor cuts that to 2 ops per hop, and the
    caller supplies pre-allocated double buffers so nothing is allocated in the
    steady state.
    """

    def __init__(self, process_group):
        self._pg = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        if process_group is not None:
            self.send_rank = dist.get_global_rank(process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(process_group, self.recv_rank)
        self._pending = []  # FIFO of outstanding Work batches

    def send_recv_packed(self, send_buf: torch.Tensor, recv_buf: torch.Tensor):
        """Ship ``send_buf`` to the next rank, receive into ``recv_buf`` from the
        previous rank. Async; caller must :meth:`wait` before touching recv_buf."""
        ops = [
            dist.P2POp(dist.isend, send_buf, self.send_rank, group=self._pg),
            dist.P2POp(dist.irecv, recv_buf, self.recv_rank, group=self._pg),
        ]
        self._pending.append(dist.batch_isend_irecv(ops))

    def wait(self):
        if self._pending:
            for req in self._pending.pop(0):
                req.wait()


class _ForwardWorkspace:
    """Reusable scratch pool keyed by (q shape, k shape, dtype, device).

    Holds the packed-KV recv/send buffers and the fp32 output/LSE accumulators.
    Diffusion runs the same attention shape thousands of times per generation,
    so a single cached workspace turns per-call allocation into zero-alloc.
    The returned ``out`` is intentionally NOT cached (allocated fresh per call)
    so it is always safe to hand back to the caller.
    """

    __slots__ = (
        "kv_bufs", "k_bufs", "v_bufs", "kv_send",
        "out_acc", "lse_acc",
        "q_shape", "kv_shape", "dtype", "device",
    )

    def __init__(self, q: torch.Tensor, k: torch.Tensor):
        b, s, hq, d = q.shape
        self.q_shape, self.kv_shape = tuple(q.shape), tuple(k.shape)
        self.dtype, self.device = q.dtype, q.device
        packed_shape = (2,) + tuple(k.shape)  # [2, B, S, H_kv, D]
        self.kv_bufs = [
            torch.empty(packed_shape, device=q.device, dtype=k.dtype),
            torch.empty(packed_shape, device=q.device, dtype=k.dtype),
        ]
        self.k_bufs = [self.kv_bufs[0][0], self.kv_bufs[1][0]]
        self.v_bufs = [self.kv_bufs[0][1], self.kv_bufs[1][1]]
        self.kv_send = torch.empty(packed_shape, device=q.device, dtype=k.dtype)
        self.out_acc = torch.empty((b, s, hq, d), device=q.device, dtype=torch.float32)
        self.lse_acc = torch.empty((b, s, hq, 1), device=q.device, dtype=torch.float32)

    def matches(self, q: torch.Tensor, k: torch.Tensor) -> bool:
        return (
            self.q_shape == tuple(q.shape)
            and self.kv_shape == tuple(k.shape)
            and self.dtype == q.dtype
            and self.device == q.device
        )


_FWD_WS: _ForwardWorkspace | None = None


def _get_fwd_ws(q: torch.Tensor, k: torch.Tensor) -> _ForwardWorkspace:
    global _FWD_WS
    if _FWD_WS is None or not _FWD_WS.matches(q, k):
        _FWD_WS = _ForwardWorkspace(q, k)
    return _FWD_WS


def ring_flash_attn_forward_opt(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
):
    """Optimized forward: packed-KV ring comm + fused Triton merge + workspace.

    Drop-in for :func:`ring_flash_attn_forward` with identical outputs. Falls
    back to the baseline when Triton is unavailable or the backend produces no
    mergeable LSE (``SPARSE_SAGE``).
    """
    # Same validation as the baseline.
    if causal and joint_tensor_key is not None and joint_strategy == "rear":
        raise ValueError(
            "joint_strategy='rear' is not compatible with causal=True in Ring Attention. "
            "When using causal attention with joint tokens, use joint_strategy='front' "
            "to ensure joint tokens act as a visible prefix for all local tokens. "
            "With 'rear' strategy, the causal mask would incorrectly block local tokens "
            "from seeing the joint tokens."
        )

    # Delegate to the baseline when the fast path does not apply.
    if not HAS_TRITON_MERGE or attn_type == AttnType.SPARSE_SAGE:
        return ring_flash_attn_forward(
            process_group,
            q,
            k,
            v,
            softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            attn_type=attn_type,
            attn_processor=attn_processor,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    comm = DoubleBufRingComm(process_group)
    batch, local_seq, num_q_heads, head_dim = q.shape

    ws = _get_fwd_ws(q, k)
    out_acc = ws.out_acc
    lse_acc = ws.lse_acc
    # Fresh output tensor: the accumulators are internal/reused, but ``out`` is
    # returned to the caller so it must not alias reused scratch.
    out = torch.empty_like(q)

    # Select the attention impl once (baseline re-selects every step).
    fn = select_flash_attn_impl(attn_type, stage="fwd-only", attn_processor=attn_processor)

    if comm.world_size > 1:
        ws.kv_send[0].copy_(k)
        ws.kv_send[1].copy_(v)
    kv_send = ws.kv_send
    k_cur, v_cur = k, v

    last_valid_step = comm.rank if causal else comm.world_size - 1
    valid_idx = 0

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            comm.send_recv_packed(kv_send, ws.kv_bufs[step & 1])

        if not causal or step <= comm.rank:
            step_k = k_cur
            step_v = v_cur
            if step == 0 and joint_tensor_key is not None:
                if joint_strategy == "front":
                    step_k = torch.cat([joint_tensor_key, step_k], dim=1)
                    step_v = torch.cat([joint_tensor_value, step_v], dim=1)
                else:
                    step_k = torch.cat([step_k, joint_tensor_key], dim=1)
                    step_v = torch.cat([step_v, joint_tensor_value], dim=1)

            block_out, block_lse = fn(
                q,
                step_k,
                step_v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )

            # Fused kernel assumes block_lse is (B, H_q, S_q) — the layout every
            # non-sparse backend (FA/FA3/FlashInfer/torch/aiter) produces. Some
            # backends (e.g. torch SDPA's efficient kernel) tail-pad S_q to an
            # alignment; the valid entries are the first `local_seq`.
            assert (
                block_lse is not None
                and block_lse.dim() == 3
                and block_lse.shape[0] == batch
                and block_lse.shape[1] == num_q_heads
                and block_lse.shape[2] >= local_seq
            ), (
                "fused ring merge expects block_lse of shape (B, Hq, Sq>=local_seq); "
                f"got {None if block_lse is None else tuple(block_lse.shape)}"
            )
            if block_lse.shape[2] != local_seq:
                block_lse = block_lse[:, :, :local_seq]

            fused_merge(
                out_acc,
                lse_acc,
                out,
                block_out,
                block_lse,
                is_first=(valid_idx == 0),
                is_final=(step == last_valid_step),
            )
            valid_idx += 1

        if step + 1 != comm.world_size:
            comm.wait()
            kv_send = ws.kv_bufs[step & 1]
            k_cur, v_cur = ws.k_bufs[step & 1], ws.v_bufs[step & 1]

    # Match the baseline return contract: lse as (B, H_q, S).
    lse = lse_acc.squeeze(dim=-1).transpose(1, 2)
    return out, lse


class RingFlashAttnFuncOpt(torch.autograd.Function):
    """Optimized Ring Flash Attention (inference only, no backward).

    Same signature/semantics as :class:`RingFlashAttnFunc`, backed by
    :func:`ring_flash_attn_forward_opt`.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_type,
        attn_processor,
        joint_tensor_key=None,
        joint_tensor_value=None,
        joint_strategy="front",
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out, softmax_lse = ring_flash_attn_forward_opt(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_type=attn_type,
            attn_processor=attn_processor,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        if not return_softmax:
            return out
        # softmax_lse may be a view into the reused workspace; copy before it
        # escapes. (SPARSE_SAGE delegation may return None.)
        lse_ret = softmax_lse.contiguous() if softmax_lse is not None else None
        return out, lse_ret, None


def ring_flash_attn_qkvpacked_func_opt(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return RingFlashAttnFuncOpt.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        None,  # attn_processor
        None,  # joint_tensor_key
        None,  # joint_tensor_value
        "front",  # joint_strategy
    )


def ring_flash_attn_kvpacked_func_opt(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return RingFlashAttnFuncOpt.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        None,  # attn_processor
        None,  # joint_tensor_key
        None,  # joint_tensor_value
        "front",  # joint_strategy
    )


def ring_flash_attn_func_opt(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, None]:
    """Optimized drop-in for :func:`ring_flash_attn_func`.

    Identical arguments and return contract; uses packed-KV double-buffered ring
    comm + a fused Triton online-softmax merge + a persistent fp32 workspace.
    Transparently falls back to the baseline for ``SPARSE_SAGE`` or when Triton
    is unavailable.
    """
    return RingFlashAttnFuncOpt.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        attn_processor,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    )
