import math

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
        # The 'metrics' dict contains the results of the evaluation
        if "eval_loss" in metrics:
            perplexity = math.exp(metrics["eval_loss"])
            # Log to WandB
            wandb.log({"eval/perplexity": perplexity}, step=state.global_step)
            print(f"Validation Perplexity: {perplexity:.2f}")