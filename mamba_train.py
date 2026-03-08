import os
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from sql_eval import SQLEvalCallback, PerplexityCallback, TrainingMetricsCallback
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from common_fns import formatting_prompts_func

# 1. Configuration
# Falcon-Mamba-7B: 7B SSM model, direct peer to Mistral-7B for a fair comparison
model_id = "tiiuae/falcon-mamba-7b"
dataset_name = "trl-lab/SQaLe-text-to-SQL-dataset"
output_dir = "./mamba-sqale-finetuned"
project_name = "mistral-7b-sql-finetuning"

run_id_path = ".run-id"
if not Path(run_id_path).exists():
    run_id = os.environ.get("run_id", datetime.now().strftime("%Y%m%d%H%M"))
    with open(run_id_path, "w") as f:
        f.write(run_id)
with open(run_id_path, "r") as f:
    run_id = f.read()
dev_run = bool(os.environ.get('DEV_RUN'))
run_type = "dev" if dev_run else "prod"

lr = 5e-5
lora_r = 16
n_distractors = 5
batch_size = 8
gradient_accumulation_steps = 4
max_length = 2048

if dev_run:
    print("=== DEV RUN MODE ===")
    print("max_steps=200, eval_steps=50, logging_steps=10, DEBUG_LENGTHS=1")

if not (wandb_key := os.getenv("WANDB_API_KEY")):
    raise Exception("no WANDB_API_KEY")

if not (hf_token := os.getenv("HF_TOKEN")):
    raise Exception("HF_TOKEN not found. Gated models may not load.")

wandb.init(
    project=project_name,
    name=f"falcon-mamba-sqale-{run_id}-{run_type}",
    config={
        "model": model_id,
        "lora_r": lora_r,
        "lr": lr,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "max_length": max_length,
        "n_distractors": n_distractors,
        # Mamba doesn't use attention; bf16 full precision used instead of 4-bit BnB
        # (bitsandbytes 4-bit quant has partial/untested support for SSM architectures)
        "quantization": "bf16",
        "attn_implementation": "none",
        "packing": False,
    }
)

# 2. Load Dataset (identical split to gpt_train.py for fair comparison)
raw_datasets = load_dataset(dataset_name, token=hf_token)
split_datasets = raw_datasets["train"].train_test_split(test_size=0.05, seed=42)

train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]
eval_dataset = eval_dataset.select(range(50))  # just use 50 examples for speed

# 3. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. Load Model
# Mamba doesn't use attention so no attn_implementation or flash-attn.
# SSM kernels from mamba-ssm serve the equivalent role and load automatically.
# bf16 full precision — BnB 4-bit quant is not reliably supported for SSM architectures.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
    token=hf_token,
)

# 5. LoRA Config for Mamba
# Mamba has no attention blocks; target the SSM projections instead.
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=32,
    target_modules=["in_proj", "out_proj", "x_proj", "embeddings"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 6. Training Arguments (identical to gpt_train.py except no flash-attn flags)
training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim="paged_adamw_32bit",
    learning_rate=lr,
    lr_scheduler_type="cosine",
    logging_steps=10,
    num_train_epochs=1,
    max_steps=200 if dev_run else -1,
    fp16=False,
    bf16=True,
    push_to_hub=True,
    report_to="wandb",
    hub_strategy="all_checkpoints",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=10,
    max_length=max_length,
    packing=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataset_num_proc=20,
    # eval
    eval_strategy="steps",
    eval_steps=50 if dev_run else 200,
    per_device_eval_batch_size=8,
    gradient_checkpointing=False,  # Not supported by FalconMamba; SSM is already memory-efficient
)

# 7. Initialize Trainer
sql_callback = SQLEvalCallback(model, tokenizer, eval_dataset, formatting_prompts_func)
perplexity_callback = PerplexityCallback()
metrics_callback = TrainingMetricsCallback(batch_size, gradient_accumulation_steps, max_length)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    callbacks=[sql_callback, perplexity_callback, metrics_callback],
    args=training_arguments,
)

# debug — always runs in DEV_RUN mode; otherwise set DEBUG_LENGTHS=1
if dev_run or os.getenv("DEBUG_LENGTHS"):
    lengths = [len(tokenizer.encode(formatting_prompts_func(x))) for x in train_dataset.select(range(200))]
    print(f"max: {max(lengths)}")
    print(f"mean: {int(sum(lengths)/len(lengths))}")
    print(f"p95: {sorted(lengths)[int(len(lengths)*0.95)]}")

# 8. Start Training
print("Checking for checkpoints...")
last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None

print(f"GPU memory before training: {torch.cuda.memory_allocated()/1e9:.1f}GB")
print(f"Starting training (Resume: {last_checkpoint})...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# 9. Save the adapter
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
