import os
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from sql_eval import SQLEvalCallback, PerplexityCallback
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from common_fns import formatting_prompts_func

# 1. Configuration
model_id = "mistralai/Mistral-7B-v0.3" # Excellent base model for SQL tasks
dataset_name = "trl-lab/SQaLe-text-to-SQL-dataset"
output_dir = "./mistral-sqale-finetuned"
project_name = "mistral-7b-sql-finetuning"

run_id_path = ".run-id"
if not Path(run_id_path).exists():
    run_id = os.environ.get("run_id", datetime.now().strftime("%Y%m%d%H%M"))
    with open(run_id_path, "w") as f:
        f.write(run_id)
with open(run_id_path, "r") as f:
    run_id = f.read()
run_type = os.environ.get('dev', "prod")

lr = 5e-5
lora_r = 16

if not (wandb_key := os.getenv("WANDB_API_KEY")):
    raise Exception("no WANDB_API_KEY")

if not (hf_token := os.getenv("HF_TOKEN")):
    raise Exception("HF_TOKEN not found. Gated models may not load.")

wandb.init(
    project=project_name, 
    name=f"mistral-v0.3-sqale-{run_id}-{run_type}",
    config={
        "lora_r": lora_r,
        "model": model_id,
    }
)

# 2. Load Dataset
raw_datasets = load_dataset(dataset_name, token=hf_token)
split_datasets = raw_datasets["train"].train_test_split(test_size=0.05, seed=42)

train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# 5. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. Load Model with 4-bit Quantization (to fit on 1 GPU)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map={"": 0}, # This forces EVERYTHING onto GPU 0
    # device_map="auto",
    attn_implementation="sdpa",  # using this over flash-attention-2 because of install headaches.
    trust_remote_code=True,
    token=hf_token,
)
model.config.use_cache = False # Required for training

# 6. LoRA Configuration (PEFT)
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 7. Training Arguments
training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=lr,
    lr_scheduler_type="cosine",
    logging_steps=10,
    num_train_epochs=1, # Adjust based on data size/time
    max_steps=-1,
    fp16=False,
    bf16=True, # Use bf16 if using A100/3090/4090
    push_to_hub=True,
    report_to="wandb",
    hub_strategy="checkpoint",  # Pushes the latest checkpoint to HF
    save_strategy="steps",      # Or "epoch"
    save_steps=100,             # Save every 100 steps
    save_total_limit=2,         # Only keep the 2 most recent checkpoints
    max_length=2048, # Adjust based on how large your schemas are
    packing=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataset_num_proc=20,
    # eval
    eval_strategy="steps",
    eval_steps=50,
    per_device_eval_batch_size=8,
    gradient_checkpointing=True,
)

# 8. Initialize Trainer
sql_callback = SQLEvalCallback(model, tokenizer, eval_dataset, formatting_prompts_func)
perplexity_callback = PerplexityCallback()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    callbacks=[sql_callback, perplexity_callback],
    args=training_arguments,
)

# debug
lengths = [len(tokenizer.encode(formatting_prompts_func(x))) for x in train_dataset.select(range(200))]
print(f"max: {max(lengths)}")
print(f"mean: {int(sum(lengths)/len(lengths))}")
print(f"p95: {sorted(lengths)[int(len(lengths)*0.95)]}")

# 9. Start Training
print("Checking for checkpoints...")
last_checkpoint = None
if os.path.exists(output_dir) and os.listdir(output_dir):
    last_checkpoint = True # Trainer will find the latest one automatically

print(f"GPU memory before training: {torch.cuda.memory_allocated()/1e9:.1f}GB")
print(f"Starting training (Resume: {last_checkpoint})...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# 10. Save the adapter
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")