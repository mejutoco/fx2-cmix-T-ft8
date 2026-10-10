"""Fine-tune the shipped quantized transformer on a (re)generated training set.

GENERATED from upstream training_recipes/cooldown.py (fx2-cmix-transformer-v1 @ 83f2603) by
v16/trackC/pod -- the TransformerConfig / quantization block below is copied VERBATIM, so the
architecture cannot drift from the weights it loads. What differs from cooldown.py:

  * starts from models/6m-q4-fp32.tch (the final shipped checkpoint), not 6m-q4.tch
  * epochs / learning rate / data dir / output come from the environment
  * runs on any GPU count dividing 8 (gradient accumulation keeps the effective batch)
  * the training loop is upstream's pysrc/train.py loop plus a checkpoint at every epoch
    boundary -- upstream saves only at the very end, which a capped or pre-empted pod cannot afford
  * no wandb; loss goes to stdout and logs/finetune.jsonl

env: FX2T_FT_EPOCHS (8)  FX2T_FT_LR (1e-6)  FX2T_DATA_DIR (data)  FX2T_FT_INIT (models/6m-q4-fp32.tch)
     FX2T_FT_OUT (models/6m-q4-fp32-ft.tch)  FX2T_FT_CKPT_DIR (models/ft)
"""
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
from os import makedirs
import os
import sys
from math import cos, pi
from dotenv import load_dotenv

from pysrc.model import Transformer, TransformerConfig, MatmulQuantization
from pysrc.quantization import Quantization
from pysrc.optimizer import make_optimizers
from pysrc.logger import Logger
from pysrc.optimizer import AdamWConfig, MuonConfig, OptimizerConfig
from pysrc.data import CollatedArticlesDataLoader
from pysrc.save_outputs import save_transformer_outputs
from pysrc.logger import LoggerConfig


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main_process: bool = rank == 0
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    # Upstream ran 8 GPUs with one 2**17-token micro-batch each per step. Keep the
    # per-GPU micro-batch and make up the same effective batch with gradient
    # accumulation, which upstream's README says reproduces the behaviour exactly.
    assert 8 % world_size == 0, "FX2T_GPU_COUNT must divide 8"
    micro_batch_tokens: int = 2**17
    gradient_accumulation_steps: int = 8 // world_size
    max_article_tokens: int | None = 2**17

    quantization_block_size: int | None = None

    weight_quantization = Quantization(
        buckets=15,
        block_size=quantization_block_size,
        scale_dtype=torch.bfloat16,
        arithmetic_dtype=torch.float32,
        scale_init="scaled_max",
        epsilon=None,
    )
    activation_quantization = Quantization(
        buckets=256,
        block_size=quantization_block_size,
        scale_dtype=torch.bfloat16,
        arithmetic_dtype=torch.float32,
        scale_init="scaled_mean",
        epsilon=None,
    )
    matmul_quantization = MatmulQuantization(
        weight=weight_quantization, activation=activation_quantization
    )

    raw_model: Transformer = Transformer(
        TransformerConfig(
            vocabulary_size=205,
            prior_embedding=True,
            prior_logit_mixing=False,
            max_variable_sequence_length=None,
            n_layers=12,
            attention_weight_sharing_pattern=list(range(12)),
            mlp_weight_sharing_pattern=list(range(12)),
            d_model=192,
            kimi_linear=[True, True, True, False] * 3,
            window_size=1024,
            window_size_multipliers=None,
            n_query_heads=3,
            n_key_value_heads=3,
            d_head=64,
            d_mla=None,
            rope_base=10_000,
            half_truncate_rope=False,
            query_key_norm=True,
            query_key_norm_gain=False,
            kimi_linear_d_head=64,
            kimi_linear_n_heads=3,
            kimi_linear_convolution_size=4,
            value_embedding_pattern=None,
            prior_value_embedding=False,
            attention_scale=None,
            d_mlp=768,
            activation_function="relu_squared",
            embedding_norm=True,
            prior_embedding_norm=True,
            skip_connections=True,
            token_embedding_connections=True,
            prior_embedding_connections=False,
            dropout=0.0,
            logit_softcap=15.0,
            prior_logprob_cap=-12.0,
            prior_embedding_on_logprobs=False,
            gradient_checkpointing=True,
            embedding_init="normal",
            prior_embedding_init="normal",
            up_init="kaiming_uniform",
            down_init="kaiming_uniform",
            unembedding_init="kaiming_uniform",
            convolution_init="kaiming_uniform",
            logit_mixing_head_init="zero",
            activation_dtype=torch.float32,
            attention_dtype=torch.float32,
            logit_dtype=torch.float32,
            weight_dtype=torch.float32,
            mlp_quantization=matmul_quantization,
            attention_full_rank_quantization=matmul_quantization,
            attention_low_rank_quantization=matmul_quantization,
            qkv_quantization=activation_quantization,
            kimi_linear_full_rank_quantization=matmul_quantization,
            kimi_linear_low_rank_quantization=matmul_quantization,
            kimi_linear_beta_projection_quantization=None,
            kimi_linear_convolution_quantization=None,
            embedding_quantization=weight_quantization,
            prior_embedding_quantization=matmul_quantization,
            unembedding_quantization=matmul_quantization,
            logit_mixing_head_quantization=None,
            adamw_quantization_scales=True,
            attention_implementation="flex_attention",
        )
    )

    raw_model = raw_model.to(device)
    init_path = os.environ.get("FX2T_FT_INIT", "models/6m-q4-fp32.tch")
    raw_model.load_state_dict(torch.load(init_path))
    model: DDP = DDP(raw_model)
    # important: do not torch compile the model as it removes the .to(torch.bfloat16).to(torch.float32) conversion applied to the quantization scales

    learning_rate: float = float(os.environ.get("FX2T_FT_LR", "1e-6"))
    epochs: int = int(os.environ.get("FX2T_FT_EPOCHS", "8"))
    final_learning_rate_multiplier: float = 1e-2
    data_dir = os.environ.get("FX2T_DATA_DIR", "data")
    out_path = os.environ.get("FX2T_FT_OUT", "models/6m-q4-fp32-ft.tch")
    ckpt_dir = os.environ.get("FX2T_FT_CKPT_DIR", "models/ft")

    data = CollatedArticlesDataLoader(
        tokens_filename=f"{data_dir}/ppmd-bytes.uint8",
        priors_filename=f"{data_dir}/ppmd-probs.float16",
        article_boundaries_filename=f"{data_dir}/article-boundaries.int32",
        mmap=True,
        epochs=epochs,
        micro_batch_tokens=micro_batch_tokens,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_article_tokens=max_article_tokens,
        max_merge_tokens=None,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    def learning_rate_multiplier_schedule(fraction_done: float) -> float:
        # cosine from 1 down to final_learning_rate_multiplier, no warmup: the scales and
        # weights are already trained, this only adapts them to the new stream
        t = (cos(fraction_done * pi) + 1) / 2
        return t + (1 - t) * final_learning_rate_multiplier

    optimizers = make_optimizers(
        raw_model,
        OptimizerConfig(
            learning_rate_multiplier_schedule=learning_rate_multiplier_schedule,
            second_order_config=MuonConfig(
                base_learning_rate=learning_rate,
                weight_decay=0.1,
                momentum=0.95,
                adjust_lr_fn="match_rms_adamw",
            ),
            adamw_config=AdamWConfig(
                base_learning_rate=learning_rate,
                weight_decay=0.01,
                betas=(0.95, 0.99),
            ),
        ),
    )
    if main_process:
        makedirs("logs", exist_ok=True)
        makedirs(ckpt_dir, exist_ok=True)
        print(f"fine-tune: init {init_path}  data {data_dir}  epochs {epochs}  lr {learning_rate}  "
              f"gpus {world_size}  micro-batch {micro_batch_tokens}  accumulation {gradient_accumulation_steps}",
              flush=True)
    logger = Logger(
        model=raw_model,
        cfg=LoggerConfig(log_to_stdout=True, file="logs/finetune.jsonl" if main_process else None),
        rank=rank,
        world_size=world_size,
    )

    # --- upstream's pysrc/train.py loop, plus a checkpoint at each epoch boundary ---
    model.train()
    last_saved_epoch = 0
    for batch in data:
        learning_rates = optimizers.update_learning_rate_schedule(fraction_done=batch.fraction_done)
        total_step_loss = 0.0
        total_step_unmasked_tokens = 0
        total_step_tokens = 0
        for micro_batch in batch.micro_batches:
            loss = model(
                input_tokens=micro_batch.input_tokens,
                priors=micro_batch.priors,
                loss_mask=micro_batch.loss_mask,
                output_tokens=micro_batch.output_tokens,
                cumulative_sequence_lengths=micro_batch.cumulative_sequence_lengths,
                override_window_size=None,
            )
            total_step_loss += loss.detach()
            total_step_unmasked_tokens += micro_batch.loss_mask.long().sum()
            loss = loss / data.gradient_accumulation_steps
            total_step_tokens += micro_batch.input_tokens.numel()
            loss.backward()
        optimizers.step()
        optimizers.zero_grad()
        logger.train_step(
            epoch=batch.epoch,
            loss=(total_step_loss / total_step_unmasked_tokens).item(),
            n_tokens=total_step_tokens,
            learning_rates=learning_rates,
            last_step=batch.is_last,
        )
        done_epochs = int(batch.epoch + 1e-9)
        if main_process and done_epochs > last_saved_epoch:
            last_saved_epoch = done_epochs
            torch.save(raw_model.state_dict(), f"{ckpt_dir}/epoch-{done_epochs:03d}.tch")
            print(f"checkpoint: {ckpt_dir}/epoch-{done_epochs:03d}.tch", flush=True)

    if main_process:
        makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(raw_model.state_dict(), out_path)
        print(f"saved {out_path}", flush=True)
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
              f"(micro-batch {micro_batch_tokens:,} tokens)", flush=True)

    dist.destroy_process_group()
    # Exit without running C++ static destructors: on torch 2.12.1+cu130 the process aborted AFTER
    # all results were written ("terminate called without an active exception", SIGABRT, pod
    # lqg8sdsdneh6qp 2026-10-05 -- 1 of 3 real runs), and the non-zero exit cost the whole pipeline.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    load_dotenv()
    main()
