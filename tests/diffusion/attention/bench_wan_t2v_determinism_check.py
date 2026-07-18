# SPDX-License-Identifier: Apache-2.0
"""Determinism control for bench_wan_t2v_ring_opt.py.

The A/B bench flagged the optimized video as differing from baseline (PSNR ~28 dB).
The ring-opt kernel matches baseline to ~2.4e-4 in the isolated micro-benchmark, so
the suspicion is that this is chaotic amplification of bf16 reduction-order noise over
the 25 denoising steps, NOT a correctness bug. This probe settles it: load ONE engine
(fixed ring-flash setting) and run the SAME seeded generation twice, then report the
frame diff. If two identical runs already diverge by a comparable amount, the A/B diff
is inherent engine nondeterminism, not the optimization.
"""
import argparse
import glob
import os
import site
import sys


def _ensure_cu13():
    dirs = []
    compat = os.path.join(sys.prefix, "cuda13-compat")
    if glob.glob(os.path.join(compat, "libcuda.so*")):
        dirs.append(compat)
    for base in (site.getsitepackages() if hasattr(site, "getsitepackages") else []):
        d = os.path.join(base, "nvidia", "cu13", "lib")
        if glob.glob(os.path.join(d, "libcudart.so.13")):
            dirs.append(d)
    if not dirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = cur.split(":") if cur else []
    need = [d for d in dirs if d not in parts]
    if need:
        os.environ["LD_LIBRARY_PATH"] = ":".join(need + parts)
        os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    _ensure_cu13()
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Path or HF id of the Wan2.2 T2V checkpoint (e.g. Wan-AI/Wan2.2-T2V-A14B-Diffusers).")
    p.add_argument("--opt", type=int, default=0)
    p.add_argument("--ring-degree", type=int, default=4)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--seed", type=int, default=142)
    args = p.parse_args()

    os.environ["VLLM_OMNI_RING_FLASH_OPT"] = str(args.opt)

    import numpy as np
    import torch

    # reuse the frame extractor from the bench
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_wan_t2v_ring_opt import _frames_to_numpy

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(ring_degree=args.ring_degree, ulysses_degree=1, use_hsdp=False),
        enforce_eager=True,
        vae_use_tiling=True,
        init_timeout=3600,
        stage_init_timeout=3600,
    )
    prompt = {"prompt": "Two anthropomorphic cats in boxing gear fight intensely on a spotlighted stage."}

    def gen():
        sp = OmniDiffusionSamplingParams(
            height=args.height, width=args.width, num_frames=args.num_frames,
            seed=args.seed, generator=torch.Generator(device="cuda").manual_seed(args.seed),
            num_inference_steps=args.steps, guidance_scale=4.0, guidance_scale_2=3.0,
            num_outputs_per_prompt=1,
        )
        return _frames_to_numpy(omni.generate(prompt, sp))

    a = gen().astype(np.float64)
    b = gen().astype(np.float64)
    diff = np.abs(a - b)
    mse = (diff ** 2).mean()
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f"\n##### DETERMINISM CONTROL (opt={args.opt}, same seed twice) #####")
    print(f"  shape           : {tuple(int(x) for x in a.shape)}")
    print(f"  max pixel diff  : {diff.max():.0f} / 255")
    print(f"  mean pixel diff : {diff.mean():.4f}")
    print(f"  pixels differing: {(diff > 0).mean() * 100:.3f}%")
    print(f"  PSNR            : {psnr:.1f} dB")
    print(f"  -> {'DETERMINISTIC (bit-identical)' if mse == 0 else 'NONDETERMINISTIC run-to-run'}")


if __name__ == "__main__":
    main()
