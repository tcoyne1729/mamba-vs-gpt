# CLAUDE.md — mamba-vs-gpt

This file gives context for AI assistants working on this codebase.

## Project Goal

Fine-tune and compare two architectures on a text-to-SQL task:
- **GPT-style** (Mistral-7B-v0.3 with LoRA)
- **Mamba** (SSM-based, to be implemented)

The end goal is a blog post comparing the two on SQL generation quality, training speed, VRAM usage, and cost efficiency.

## Status

**Work in progress. Many known issues.** See `mamba_gpt_dev_checklist.md` for the full list of bugs and missing features.

- `gpt_train.py` — functional but has several bugs (see checklist)
- `mamba_train.py` — stub, not yet implemented
- `benchmark.py` — post-training evaluation, not yet complete
- `sql_eval.py` — has bugs in prompt splitting and field names (see checklist)

## Repo Structure

```
gpt_train.py        # GPT fine-tuning script (Mistral-7B + LoRA + 4-bit quant)
mamba_train.py      # Mamba fine-tuning script (not yet implemented)
common_fns.py       # Shared formatting function used by both training scripts
sql_eval.py         # Custom wandb callbacks: SQLEvalCallback, PerplexityCallback
benchmark.py        # Post-training benchmark (runs eval_data.txt through both models)
eval_data.txt       # Synthetic test examples for benchmark
setup.sh            # Instance setup script — run via curl | bash on vast.ai
runners/            # Shell scripts to launch training runs
Makefile            # Shortcuts for common commands
pyproject.toml      # uv dependency management
```

## Environment

Runs on **vast.ai GPU instances** (currently RTX 5090, targeting A100 for final runs).
Python **3.12** managed via `uv`. Do not use 3.14 — wheel availability is poor.

Setup a fresh instance with:
```bash
curl -fsSL https://raw.githubusercontent.com/tcoyne1729/mamba-vs-gpt/main/setup.sh | bash
```

Required env vars — set in vast.ai instance config:
```
WANDB_API_KEY
HF_TOKEN
```

## Key Design Decisions

**Dataset:** `trl-lab/SQaLe-text-to-SQL-dataset` — contains very large schemas (mean ~9k tokens). We use `extract_schema_with_distractors()` in `common_fns.py` to include only relevant tables plus 5 random distractors, bringing mean token length to ~1100.

**Schema distractor approach:** The model learns to find relevant tables among noise rather than seeing a pre-filtered schema. `n_distractors=5` gives a p95 of ~2026 tokens which fits `max_length=2048`.

**Quantization:** 4-bit NF4 via bitsandbytes. Under review — may switch to bf16 full precision since VRAM headroom exists on 32GB cards.

**Checkpointing:** LoRA adapter only (~130MB per checkpoint). Pushed to HuggingFace Hub. Resume logic needs fixing (see checklist).

## Reproducibility Rules

When implementing `mamba_train.py` and final benchmark runs, the following must be identical to `gpt_train.py` for a fair comparison:
- Same dataset and `seed=42` train/eval split
- Same `formatting_prompts_func` and `n_distractors=5`
- Same `max_length=2048`
- Same eval metrics (`sql_eval.py`)
- Same wandb project — use model name as tag

## wandb

Project: `mistral-7b-sql-finetuning`
Run naming: `{model}-sqale-{run_id}-{dev|prod}`

## Known Issues

See `mamba_gpt_dev_checklist.md` for the full prioritised list. Top issues:

1. `torch_dtype` param typo in `gpt_train.py`
2. `sql_eval.py` prompt split delimiter mismatch
3. `hub_strategy="checkpoint"` overwrites previous checkpoints on HF
4. Resume from checkpoint on fresh instance doesn't work correctly
5. `mamba_train.py` not yet implemented
