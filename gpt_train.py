import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from common_fns import formatting_prompts_func
import wandb
from datetime import datetime
import os


# 1. Configuration
model_id = "mistralai/Mistral-7B-v0.3" # Excellent base model for SQL tasks
dataset_name = "trl-lab/SQaLe-text-to-SQL-dataset"
output_dir = "./mistral-sqale-finetuned"
project_name = "mistral-7b-sql-finetuning"

run_id = os.environ.get("run_id", datetime.now().strftime("%Y%m%d%H%M"))
run_type = os.environ.get('dev', "prod")
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
dataset = load_dataset(dataset_name, split="train", token=hf_token)

# 4. Load Model with 4-bit Quantization (to fit on 1 GPU)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=hf_token,
)
model.config.use_cache = False # Required for training

# 5. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 6. LoRA Configuration (PEFT)
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 7. Training Arguments
training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
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
    # packing=True,
    dataset_num_proc=min(os.cpu_count(), 32), # Use all available CPU cores
    token=hf_token,
    hub_model_id=f"tcoyne1729/{project_name}"
)

# 8. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    # tokenizer=tokenizer,
    args=training_arguments,
)

# 9. Start Training
print("Checking for checkpoints...")
last_checkpoint = None
if os.path.exists(output_dir) and os.listdir(output_dir):
    last_checkpoint = True # Trainer will find the latest one automatically

print(f"Starting training (Resume: {last_checkpoint})...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# 10. Save the adapter
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")