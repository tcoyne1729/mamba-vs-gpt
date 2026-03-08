import math
import time

import sqlglot
import torch
import wandb
from transformers import TrainerCallback


class SQLEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset, formatting_func, num_samples=50):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.formatting_func = formatting_func
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, **kwargs):
        self.model.eval()
        
        valid_sql = 0
        exact_match = 0
        total = min(self.num_samples, len(self.eval_dataset))
        
        print(f"\n--- SQL Eval (Step {state.global_step}) ---")
        
        for i in range(total):
            sample = self.eval_dataset[i]
            full_prompt = self.formatting_func(sample)
            input_text = full_prompt.split("### sql\n")[0] + "### sql\n"
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=0.1,  # near-greedy for eval
                    do_sample=False,
                )
            
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_sql = prediction.split("### sql\n")[-1].strip()
            expected_sql = sample['query'].strip()
            
            # 1. Syntax validity via sqlglot
            try:
                sqlglot.parse(generated_sql)
                valid_sql += 1
            except:
                pass
            
            # 2. Exact match (normalised)
            if generated_sql.lower().strip() == expected_sql.lower().strip():
                exact_match += 1
            
            if i == 0:
                print(f"Expected: {expected_sql}")
                print(f"Generated: {generated_sql}\n")
        
        validity_rate = valid_sql / total
        exact_match_rate = exact_match / total
        
        wandb.log({
            "eval/sql_validity_rate": validity_rate,
            "eval/exact_match_rate": exact_match_rate,
            "eval/num_samples": total,
        }, step=state.global_step)
        
        print(f"Validity: {validity_rate*100:.1f}% | Exact Match: {exact_match_rate*100:.1f}%")
        self.model.train()

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            perplexity = math.exp(metrics["eval_loss"])
            wandb.log({"eval/perplexity": perplexity}, step=state.global_step)
            print(f"Validation Perplexity: {perplexity:.2f}")


class TrainingMetricsCallback(TrainerCallback):
    """Logs VRAM usage, tokens/sec throughput, and wall-clock time to wandb."""

    def __init__(self, batch_size: int, gradient_accumulation_steps: int, max_length: int):
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self._step_start_time: float | None = None
        self._train_start_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start_time = time.time()
        wandb.run.summary["train_start_timestamp"] = self._train_start_time

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        metrics = {}

        # VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            metrics["train/vram_gb"] = vram_gb

        # Tokens/sec — estimated from step time and effective tokens per step
        if self._step_start_time is not None:
            elapsed = time.time() - self._step_start_time
            if elapsed > 0:
                tokens_per_step = self.batch_size * self.gradient_accumulation_steps * self.max_length
                metrics["train/tokens_per_sec"] = tokens_per_step / elapsed

        # Wall-clock elapsed
        if self._train_start_time is not None:
            metrics["train/elapsed_hours"] = (time.time() - self._train_start_time) / 3600

        if metrics:
            wandb.log(metrics, step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if self._train_start_time is not None:
            elapsed_h = (time.time() - self._train_start_time) / 3600
            wandb.run.summary[f"checkpoint_step_{state.global_step}_elapsed_hours"] = elapsed_h