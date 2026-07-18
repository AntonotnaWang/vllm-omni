# Optimized Ring-Flash-Attention for diffusion sequence parallelism

## Summary

This PR adds an optimized ring-attention forward path (`ring_flash_attn_func_opt`)
for the diffusion sequence-parallel (ring / USP) attention used by Wan, Qwen-Image,
Flux-style DiTs, etc. It is a **faithful drop-in** for the existing
`ring_flash_attn_func` — identical signature, identical output up to bf16
reduction-order noise — that removes per-ring-step overhead in the KV
communication and the online-softmax merge. It is now **enabled by default**
(`VLLM_OMNI_RING_FLASH_OPT=1`); set the env var to `0` to fall back to the
original path for A/B comparison.

## Problem

Ring attention shards the sequence across `ring_degree` GPUs and, at each of the
`ring_degree` steps, (a) sends/receives one rank's K/V shard and (b) merges the
partial attention output into a running accumulator via an online softmax. The
attention math itself is a FlashAttention kernel, but the *surrounding* per-step
work is not free:

- **KV communication** issued K and V as **two separate** `send_recv` transfers
  per step, with no explicit double buffering.
- **Online-softmax merge** (`ring_utils.update_out_and_lse`) ran as **several
  elementwise kernels** (`sigmoid`/`logsigmoid`/`sub`/`mul`) plus a `bf16 <-> fp32`
  round-trip on every step.
- **Per-step allocation**: fresh fp32 accumulators were created, and the block
  attention implementation was re-selected, every step.

For long sequences this overhead is a real fraction of DiT time. The lever is the
**per-GPU (local) shard length**: attention compute is `O(local_seq x global_seq)`
while the comm+merge the optimization targets is `O(global_seq)`, so the relative
overhead — and therefore the win — grows as the local shard shrinks (higher ring
degree, longer video/image sequence split across more GPUs).

## What the optimization does

`ring_flash_attn_func_opt` combines three techniques (all inference / forward-only):

1. **Packed-KV double-buffered ring comm** (`DoubleBufRingComm`): K and V are
   packed into a single contiguous buffer and exchanged in **one** `send_recv`
   per step (half the NCCL ops); two ping-pong buffers let the next step's
   transfer overlap the current step's attention compute.
2. **Fused Triton online-softmax merge** (`ring/ring_triton.py::fused_merge`):
   the whole per-step merge collapses into **one** Triton kernel launch that
   works in place on the fp32 accumulators and writes the **final** step
   straight to the bf16 output. Numerics are identical to the baseline
   `update_out_and_lse` (`out = out*(1-s) + block_out*s`, `s = sigmoid(block_lse - lse)`).
3. **Persistent fp32 workspace** (`_ForwardWorkspace` / `_get_fwd_ws`): reusable
   accumulators/comm buffers keyed by tensor shape, and the block attention impl
   is selected **once** instead of every step.

The optimized path transparently **falls back to the baseline** for
`SPARSE_SAGE` or when Triton is unavailable, so behavior is never worse than
before.

## Code added

Core (2 files changed, 1 new module):

| File | Change | Contents |
|---|---|---|
| `vllm_omni/diffusion/attention/backends/ring_flash_attn.py` | +432 | `ring_flash_attn_func_opt` (+ `qkvpacked`/`kvpacked` variants), `RingFlashAttnFuncOpt` (autograd fn), `ring_flash_attn_forward_opt` (the optimized loop), `DoubleBufRingComm`, `_ForwardWorkspace` / `_get_fwd_ws` |
| `vllm_omni/diffusion/attention/backends/ring/ring_triton.py` | +138 (new) | `_ring_merge_kernel` (`@triton.jit`), `fused_merge`, `HAS_TRITON_MERGE` guard/fallback |
| `vllm_omni/diffusion/attention/parallel/ring.py` | +18/-2 | dispatch on `VLLM_OMNI_RING_FLASH_OPT` (default **on**) |

The baseline path (`ring_flash_attn_func`, `ring_utils.update_out_and_lse`) is
**untouched** — the change is purely additive and opt-out-able.

Benchmarks / tests (new, under `tests/diffusion/attention/`):

| File | Purpose |
|---|---|
| `bench_ring_flash_attn_opt.py` | Correctness (opt vs baseline vs fp32 reference) + speed across shapes and parameters; exits non-zero on any FAIL |
| `bench_merge_micro.py` | Micro-benchmark of the fused merge alone |
| `bench_wan_t2v_ring_opt.py` | End-to-end Wan2.2 text-to-video A/B (baseline vs opt), correctness + speed |
| `bench_wan_t2v_determinism_check.py` | Determinism control (same seed twice, one engine) to attribute end-to-end frame diffs to bf16 nondeterminism rather than the kernel |

## Dependencies

**No new package dependency.**

- The fused merge uses **`triton`**, which is **already present** — it ships as a
  dependency of `torch` (Linux + CUDA) and is already imported by several
  existing modules (`fish_kvcache_triton.py`, `snake_activation.py`, etc.). It is
  not declared explicitly in `requirements/`, and this PR does not need to add it.
- The per-block attention kernel (FA3 / FA2 / torch-SDPA / AITER) is the **same**
  backend the baseline ring path already uses — no change. The torch-SDPA backend
  requires no FlashAttention at all, so the optimization is exercisable anywhere.

Environment note (not a Python package): FA3 (`fa3-fwd`) is a CUDA-13 build. On a
host whose system driver predates CUDA 13, a `cuda-compat-13-0` forward-compat
driver must be on `LD_LIBRARY_PATH`. The benchmarks auto-add `<venv>/cuda13-compat`
and `nvidia/cu13/lib` if present, and no-op otherwise.

## Correctness

Validated by `bench_ring_flash_attn_opt.py` (FA3 backend, H100, ring degree 2 and 4).
Two properties are checked per case:

1. **opt == baseline** — the optimization is a faithful drop-in. Holds for every
   shape and parameter.
2. **opt / baseline == fp32 full-attention reference** — the whole ring path is
   correct (checked where the backend honors the parameter).

Results (max abs diff on bf16 outputs; tolerances opt-vs-base `2e-2`, vs-ref `5e-2`):

| Check | Observed range | Tolerance | Verdict |
|---|---|---|---|
| opt vs baseline (all shapes & params) | `1.2e-4 - 4.9e-4` | `2e-2` | PASS |
| opt/baseline vs fp32 reference (non-causal) | `2.4e-4 - 9.3e-4` | `5e-2` | PASS |
| opt/baseline vs fp32 reference (causal) | `7.4e-3 - 9.3e-3` | `5e-2` | PASS |

Coverage:

- **Shapes**: `seq_len in {2048, 8192}` (local per rank) x `head_dim in {64, 128, 256}`
  x `{MHA 16/16, GQA 16/4}` x `{non-causal, causal}` — all PASS.
- **Parameters**: `softmax_scale`, `softcap`, `window_size` (sliding), `alibi_slopes`,
  `dropout_p`, and `joint` (text-conditioning) tokens.
  - `softmax_scale`, `softcap`, `joint`, `dropout_p`: PASS.
  - `window_size`, `alibi_slopes`: **SKIP** on the FA3 build used here — that build
    is compiled with `DISABLE_LOCAL` (no sliding window) and does not forward
    alibi. This is a backend-capability gap, not an opt bug; the harness detects it
    and skips (it would otherwise compare against a reference the kernel ignores).
    Both are exercised on FA2/AITER backends.

The `opt == baseline` check is backend-independent and passes for every case,
which is the property that matters: the optimization changes only the ring
comm/merge, never the per-block attention result.

## Performance

**Isolated attention** (`bench_ring_flash_attn_opt.py`, FA3, 4xH100, ring degree 4):

| Case | speedup |
|---|---|
| non-causal, MHA/GQA, d 64-256, s 2048-8192 | **1.09-1.21x** |
| causal, MHA, d 64-256, s 2048-8192 | **1.08-1.34x** |
| causal, GQA 16/4, s 2048 (3.78 ms -> 0.95 ms) | **~4x** (baseline hit a slow GQA-causal path) |

A dedicated local-shard sweep confirms the scaling law — the win grows as the
**local** shard shrinks:

| local seq (per GPU) | global seq | ring degree | isolated speedup |
|---|---|---|---|
| 8.2k | 32.8k | 4 | 1.22x |
| 9.45k | 75.6k | 8 | 1.20x |
| 18.9k | 75.6k | 4 | 1.06x |

**End-to-end Wan2.2-T2V-A14B** (`bench_wan_t2v_ring_opt.py`, 8xH100, 81 frames, HSDP):

| Resolution | ring degree | baseline (median) | opt (median) | speedup |
|---|---|---|---|---|
| 832x480 | 4 | 54.1 s | 48.6 s | **1.11x** |
| 1280x720 | 8 | 107.7 s | 99.5 s | **1.08x** |

End-to-end speedup is diluted by text-encode, VAE-decode, and the HSDP weight
all-gather (all identical between the two runs, so the delta is purely the ring
optimization). Practical guidance: raise `ring_degree` to keep the per-GPU shard
small — that is the lever, not resolution alone.

## How to run the benchmarks

Activate the venv first (adds the CUDA-13 compat driver to the loader path if
present). Pass the model path/id at runtime — no paths are hard-coded.

**Correctness + speed micro-benchmark** (doubles as a regression test — non-zero
exit on any FAIL):

    cd projs/learn/vllm-omni
    # ring degree == nproc_per_node; auto-selects fa3 -> fa2 -> torch-SDPA
    torchrun --nproc_per_node=4 tests/diffusion/attention/bench_ring_flash_attn_opt.py --iters 20
    # quick sweep:            add --quick
    # force a backend:        --attn-type {torch|fa|fa3|aiter}

**End-to-end Wan2.2 text-to-video A/B**:

    python tests/diffusion/attention/bench_wan_t2v_ring_opt.py \
      --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
      --ring-degree 8 --use-hsdp --steps 25 --height 720 --width 1280 --num-frames 81 \
      --flow-shift 5.0 --runs 3 --enforce-eager
    # outputs (videos, frame .npy, timing json) go under --workdir (a temp dir by default)

**Determinism control** (attributes any end-to-end frame diff to bf16
nondeterminism vs the kernel):

    python tests/diffusion/attention/bench_wan_t2v_determinism_check.py \
      --model Wan-AI/Wan2.2-T2V-A14B-Diffusers --opt 0 --ring-degree 4

## Toggle

    export VLLM_OMNI_RING_FLASH_OPT=1   # optimized ring path (default)
    export VLLM_OMNI_RING_FLASH_OPT=0   # original baseline path
