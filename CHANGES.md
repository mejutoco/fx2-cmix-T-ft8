# Changes against upstream fx2-cmix-transformer-v1 @ 83f2603

Upstream: https://github.com/astOwOlfo/fx2-cmix-transformer-v1 (Vladimir Ivanov), GPL-3.0.
This tree is **exactly the tree that built the submitted `cmix`**: upstream at commit `83f2603`,
plus the items below. `git diff 83f2603` against upstream is the complete diff.

## 1. The submitted change (the only thing that changes the compressor's behaviour)

| file | change |
|---|---|
| `models/6m-q4-fp32.tfwc2` | replaced. Upstream: 2,930,652 B, sha256 `7f4db6c8c843a7e6264b6a48ed4805e9e431f543df7a9a0ecb37a35a5e4b8860`. Here: 2,934,692 B, sha256 `82f85e263aec98f787997d39a78062322f3b8074a00c9b65535c71fe560adb89`. Made by 8 more epochs of QAT on upstream's `models/6m-q4-fp32.tch` (`training_recipes/finetune.py`), exported with `training_recipes/export_blob.py` and packed with upstream's `pysrc/weights_compress.py`. |
| `models/6m-q4-fp32-ft.tch` | added. The fine-tuned checkpoint behind the blob (23,870,167 B, sha256 `fc34f639c04c03426551c1068236a402d4e94a753e1dbcfc071f8ef81b4408fb`). Not used by the compressor. |

Upstream's other checkpoints (`models/*.tch`, 160 MB) are not copied here. Get them from upstream
if you want to re-run the training.

## 2. Inactive source additions (compiled in, never executed)

The tree carries two experiments from the same campaign. Both are behind preprocessor defines that
are **not set** in the submitted build (`EXTRA_DEFINES` is empty, so `CFLAGS_DEFINES` is upstream's
`-DSEED=923 -DUPDATE_LIMIT=3000`). They do not change what the compressor does. They do change the
binary by a few bytes, which is why we ship this tree and not pristine upstream (see §4).

| file | what | active? |
|---|---|---|
| `src/r1_reorder_transform.cpp/.h` (+624 lines) | a port of cmix-lex's regime-1 block sort, behind `#ifdef FX2T_LEX` | **no** — `FX2T_LEX` not defined; the object is linked but its code is unreachable |
| `src/runner.cpp` (+53/−4) | `#ifdef FX2T_LEX` call sites; outside the `#ifdef`, `RunCompression`/`RunDecompression` gain one defaulted `const char*` parameter which the `#else` branch discards with `(void)` | **no** |
| `src/predictor.cpp/.h` (+112) | an output-prior probe, entirely inside `#ifdef OBIAS_PROBE` | **no** — `OBIAS_PROBE` not defined |
| `makefile` (+3/−3) | compiles and links `r1_reorder_transform.o` | yes, links dead code |
| `build_and_construct_comp.sh` (+1/−1) | appends `${EXTRA_DEFINES:-}` to `CFLAGS_DEFINES` | yes, but empty |
| `cpp_infer/src/dump_weights.cpp` (+32) | a standalone tool that lists tensors in a weights blob; not part of `cmix` | not built by the script |
| `src/r1_roundtrip_test.cpp` (+32) | standalone test for the sort; not part of `cmix` | not built by the script |

## 3. Training scripts (do not affect the compressor)

| file | what |
|---|---|
| `training_recipes/finetune.py` | upstream's `cooldown.py` with: start from `6m-q4-fp32.tch`; epochs / lr / paths from the environment; gradient accumulation so 1 GPU gives upstream's 8-GPU batch; a checkpoint per epoch; no wandb. Model and quantization config are verbatim. Diff: `git diff` against upstream `training_recipes/cooldown.py`. |
| `training_recipes/export_blob.py` | builds the model from the same verbatim config, loads a checkpoint, calls upstream's `pysrc/export_weights.py:export_weights`. Needed because upstream's `export_weights.py:main()` has a stale config helper (`TypeError` on `query_key_norm_gain`). |
| `training_recipes/eval_loss.py` | forward-only loss of a checkpoint on the training data; used to measure the quantization penalty ((see the submission documents)). |

## 4. Build, and the 88-byte note

`./build_and_construct_comp.sh` unchanged in effect (clang-17, PGO+LTO, `-march=x86-64-v3
-mrecip=none`, UPX 5.1.1 as `upx-ucl`). Output `run/cmix` = 3,422,266 B, sha256
`a173a4c5d29b56e29b93554b5938a0efe8ff9528c08845bf4f6e1f0b83cc21ad`; `run/cmix_orig` = 186,420 B.

A build of **pristine** upstream `83f2603` with the same blob gives `cmix_orig` = 186,508 B and
`cmix` = 3,422,354 B (+88 B, charged twice = +176 B of S) with identical `comp_dict`, `comp_order`,
`comp_tfweights` and `header.dat` . The
difference is the inactive code in §2. We did not run the full compression with the pristine build,
so we ship the tree we measured.
