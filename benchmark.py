import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def get_gpu_memory():
    return torch.cuda.memory_reserved() / 1e9  # Convert to GB

def generate_decoys(n):
    """Generates N decoy tables to inflate schema length."""
    decoys = ""
    for i in range(n):
        decoys += f"CREATE TABLE decoy_table_{i} (col_a INT, col_b TEXT, col_c TIMESTAMP);\n"
    return decoys

def benchmark_model(model, tokenizer, schema, question, decoys_count=0):
    full_schema = generate_decoys(decoys_count) + schema
    prompt = f"<schema>\n{full_schema}\n<question>\n{question}\n<query>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_vram = get_gpu_memory()
    
    # Measure Latency
    start_time = time.time()
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=64, 
            do_sample=False,
            use_cache=True
        )
    end_time = time.time()
    
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    total_time = end_time - start_time
    gen_tokens = output_tokens.shape[1] - input_len
    tps = gen_tokens / total_time
    
    decoded_output = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True)
    
    return {
        "input_tokens": input_len,
        "peak_vram_gb": peak_vram,
        "tokens_per_sec": tps,
        "output": decoded_output.strip()
    }

# --- MAIN EXECUTION ---
models_to_test = [
    {"name": "Mistral-7B", "path": "mistralai/Mistral-7B-v0.3", "adapter": "./mistral-sqale-finetuned"},
    {"name": "Mamba-7B", "path": "tiiuae/falcon-mamba-7b", "adapter": "./mamba-sqale-finetuned"}
]

# Load test cases
with open("eval_data.txt", "r") as f:
    cases = f.read().split("===")[1:]

for model_info in models_to_test:
    print(f"\n--- Testing {model_info['name']} ---")
    
    # Load base + adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        model_info["path"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_info["adapter"])
    tokenizer = AutoTokenizer.from_pretrained(model_info["path"])

    for decoy_lvl in [0, 50, 200]: # Test with 0, 50, and 200 fake tables
        print(f"\n[Decoy Tables: {decoy_lvl}]")
        for case in cases:
            lines = case.strip().split("\n")
            schema = lines[1]
            question = lines[3]
            
            res = benchmark_model(model, tokenizer, schema, question, decoys_count=decoy_lvl)
            print(f"Tokens: {res['input_tokens']} | VRAM: {res['peak_vram_gb']:.2f}GB | TPS: {res['tokens_per_sec']:.1f}")
            print(f"SQL: {res['output']}\n")
    
    del model, base_model # Free memory for next model
    torch.cuda.empty_cache()