# fx2-cmix-T-ft8 — Hutter Prize submission

Author: Jesus Mejuto. Date: 2026-10-10. License: GPL-3.0 (see `LICENSE`).

This is [fx2-cmix-transformer-v1](https://github.com/astOwOlfo/fx2-cmix-transformer-v1)
(fx2-cmix-T, Vladimir Ivanov, commit `83f2603`) with **one changed file**: the transformer weights
`models/6m-q4-fp32.tfwc2`, trained for 8 more quantization-aware epochs from upstream's shipped
checkpoint. The compressor executes upstream's code unchanged. `CHANGES.md` lists every difference
from upstream, including inactive `#ifdef` additions that are compiled but never executed.

| | bytes | sha256 |
|---|--:|---|
| `cmix` (compressor, S1) | 3,422,266 | `a173a4c5d29b56e29b93554b5938a0efe8ff9528c08845bf4f6e1f0b83cc21ad` |
| `archive9` (self-extracting archive, S2) | 96,945,005 | `2b3892ca5a43e1b25bcbbbb5bc79f95cc09331953b324333500e61a3b45eba5b` |
| **S = S1 + S2** | **100,367,271** | |

The binaries are attached to the GitHub release of this repository, with `SHA256SUMS`.

## Decompress

Put `archive9` alone in an empty directory and run:

```bash
./archive9
```

It writes `enwik9_uncompressed` (1,000,000,000 bytes, sha256
`159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc`) and some temporary files.
Needs one x86-64 core with AVX2, under 10 GB of RAM and about 25 GB of temporary disk.
Measured: 130,005 CPU-seconds (36.1 h) on an AMD Ryzen 9 5900X, peak RSS 8,152,728 KiB under
an enforced 10 GiB limit.

## Compile

Requirements: Linux x86-64, `clang-17`, `llvm-profdata-17`, `make`, `objdump`, and UPX 5.1.1
available on `PATH` as `upx-ucl`. Then:

```bash
./build_and_construct_comp.sh
```

This is upstream's script, unchanged: PGO + LTO with `-march=x86-64-v3 -mrecip=none`, two guards
(AVX2-only; no reciprocal-estimate instructions), UPX, then the dictionary, the article order and
the weights are appended. It produces `run/cmix`. The build is deterministic: a fresh clone
reproduces the sha256 above.

## Compress

Put `run/cmix` and `enwik9` in an empty directory and run:

```bash
./cmix -e enwik9 enwik9.comp
```

It produces `archive9`. Measured: 111,424 CPU-seconds (31.0 h) on the same machine, peak RSS
9,681,988 KiB.

## Re-create the weights (optional, needs one NVIDIA GPU and ~300 GB of disk)

```bash
./make_training_data.sh                       # upstream's script: cmix --ppmd-only dump of enwik9
FX2T_FT_EPOCHS=8 FX2T_FT_LR=2.5e-6 FX2T_DATA_DIR=data OMP_NUM_THREADS=16 \
  uv run torchrun --nproc_per_node 1 -m training_recipes.finetune
uv run python -m training_recipes.export_blob models/6m-q4-fp32-ft.tch cpp_infer/data/weights-ft.bin
uv run python -m pysrc.weights_compress compress cpp_infer/data/weights-ft.bin models/6m-q4-fp32.tfwc2
uv run python -m pysrc.weights_compress verify   cpp_infer/data/weights-ft.bin models/6m-q4-fp32.tfwc2
```

`training_recipes/finetune.py` is upstream's `cooldown.py` with the start checkpoint, epochs and
learning rate taken from the environment, gradient accumulation for a single GPU, and a checkpoint
per epoch. GPU training is not bit-reproducible; the exact checkpoint behind the shipped blob is
`models/6m-q4-fp32-ft.tch`. Upstream's other checkpoints are not included here.

Upstream's original README is `README-upstream.md`; upstream's description of the method is
`writeup.md`.
