"""Checkpoint -> cpp_infer weights.bin, using the SAME verbatim config block as the fine-tune and
eval recipes (training_recipes/eval_loss.py: build_quantized).

Upstream's pysrc/export_weights.py:main() cannot run as shipped: its make_reference_config() was
not updated when TransformerConfig gained `query_key_norm_gain` (TypeError, found on a pod
2026-10-05). export_weights(model, path) itself is fine and is what this calls.

usage: python -m training_recipes.export_blob <checkpoint.tch> <weights.bin>
"""
import os
import sys

import torch

import pysrc.export_weights as ew
from training_recipes.eval_loss import build_quantized


def main() -> None:
    ckpt, out = sys.argv[1], sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_quantized(activations=os.environ.get("FX2T_EVAL_ACT", "1") == "1",
                            emb_bits=int(os.environ.get("FX2T_EVAL_EMB_BITS", "4"))).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"export: {ckpt} on {device} -> {out}", flush=True)
    ew.export_weights(model, out)


if __name__ == "__main__":
    main()
