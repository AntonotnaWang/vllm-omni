# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: fused Triton merge vs baseline update_out_and_lse.

Isolates the per-ring-step online-softmax merge (independent of the attention
backend and of comm), so the merge speedup is visible even when the per-step
attention kernel would otherwise dominate end-to-end timing.

    cd projs/learn/vllm-omni
    python tests/diffusion/attention/bench_merge_micro.py
    #   --batch 1 --heads 24 --seqlen 4096 --head-dim 128 --steps 8 --iters 200
"""

import argparse

import torch

from vllm_omni.diffusion.attention.backends.ring.ring_triton import HAS_TRITON_MERGE, fused_merge
from vllm_omni.diffusion.attention.backends.ring.ring_utils import update_out_and_lse


def _baseline_merge(block_out, block_lse, steps):
    """Baseline: torch update_out_and_lse accumulating `steps` blocks, then cast."""
    out = None
    lse = None
    for _ in range(steps):
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)
    return out.to(block_out.dtype)


def _fused_merge(out_acc, lse_acc, out, block_out, block_lse, steps):
    for step in range(steps):
        fused_merge(out_acc, lse_acc, out, block_out, block_lse, is_first=(step == 0), is_final=(step == steps - 1))
    return out


def _time(fn, iters):
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--steps", type=int, default=8, help="ring steps to accumulate")
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    assert HAS_TRITON_MERGE, "Triton not available"
    dev = "cuda"
    b, h, s, d = args.batch, args.heads, args.seqlen, args.head_dim
    torch.manual_seed(0)

    block_out = torch.randn(b, s, h, d, device=dev, dtype=torch.bfloat16)
    block_lse = torch.randn(b, h, s, device=dev, dtype=torch.float32)

    out_acc = torch.empty(b, s, h, d, device=dev, dtype=torch.float32)
    lse_acc = torch.empty(b, s, h, 1, device=dev, dtype=torch.float32)
    out = torch.empty_like(block_out)

    def base_fn():
        return _baseline_merge(block_out, block_lse, args.steps)

    def fuse_fn():
        return _fused_merge(out_acc, lse_acc, out, block_out, block_lse, args.steps)

    # correctness: same inputs -> same accumulated result
    ref = base_fn()
    got = fuse_fn()
    max_diff = (ref.float() - got.float()).abs().max().item()

    for _ in range(20):  # warmup
        base_fn()
        fuse_fn()
    t_base = _time(base_fn, args.iters)
    t_fuse = _time(fuse_fn, args.iters)

    print(f"B={b} H={h} S={s} D={d} steps={args.steps}")
    print(f"  max_abs_diff(fused vs baseline) = {max_diff:.3e}")
    print(f"  baseline update_out_and_lse : {t_base * 1e3:8.1f} us / {args.steps} merges")
    print(f"  fused triton merge          : {t_fuse * 1e3:8.1f} us / {args.steps} merges")
    print(f"  merge speedup               : {t_base / t_fuse:5.2f}x")


if __name__ == "__main__":
    main()
