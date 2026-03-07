import torch
import wandb
import math
from transformers import TrainerCallback

class SQLEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset, formatting_func, num_samples=5):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.formatting_func = formatting_func
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, **kwargs):
        self.model.eval()
        success_count = 0
        
        print(f"\n--- Running Custom SQL Eval (Step {state.global_step}) ---")
        
        for i in range(self.num_samples):
            # Grab a sample and format it
            sample = self.eval_dataset[i]
            # Use your existing formatting function
            full_prompt = self.formatting_func(sample)[0] 
            
            # We want to prompt the model with everything EXCEPT the expected SQL
            # This assumes your prompt ends with something like "SQL:"
            input_text = full_prompt.split("SQL:")[0] + "SQL:"
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=64, 
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_sql = prediction.split("SQL:")[-1].strip()
            
            # Simple heuristic check: Does it start with SELECT? 
            # (In a real run, you'd use a SQL parser like `sqlglot`)
            if "SELECT" in generated_sql.upper() and ";" in generated_sql:
                success_count += 1
                
            if i == 0: # Print the first one for a quick visual check
                print(f"Target: {sample['answer'] if 'answer' in sample else 'N/A'}")
                print(f"Generated: {generated_sql}\n")

        # Log to WandB
        valid_rate = success_count / self.num_samples
        wandb.log({"eval/sql_validity_rate": valid_rate}, step=state.global_step)
        print(f"SQL Validity Rate: {valid_rate * 100}%")
        self.model.train()

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # The 'metrics' dict contains the results of the evaluation
        if "eval_loss" in metrics:
            perplexity = math.exp(metrics["eval_loss"])
            # Log to WandB
            wandb.log({"eval/perplexity": perplexity}, step=state.global_step)
            print(f"Validation Perplexity: {perplexity:.2f}")