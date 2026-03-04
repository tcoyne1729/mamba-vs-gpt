import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from common_fns import formatting_prompts_func

# 1. Configuration
# Falcon-Mamba-7B is the perfect 7B peer for Mistral
model_id = "tiiuae/falcon-mamba-7b" 
dataset_name = "trl-lab/SQaLe-text-to-SQL-dataset"
output_dir = "./mamba-sqale-finetuned"

# 2. Dataset Formatting (Identical to Mistral script for fairness)
dataset = load_dataset(dataset_name, split="train")

# 3. Load Model
# Note: Mamba models are already quite memory efficient. 
# We use bfloat16 for high-precision comparison.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 4. LoRA Config for Mamba
# Mamba doesn't have "Attention" blocks, so we target the SSM projections
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 5. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 6. Trainer (Keep hyperparams identical to the Mistral script)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="no", # Save manually at the end
    ),
    formatting_func=formatting_prompts_func,
    max_seq_length=2048,
)

trainer.train()
trainer.save_model(output_dir)