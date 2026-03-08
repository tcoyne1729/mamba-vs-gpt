import re
import random

def formatting_prompts_func(example: dict):
    schema = extract_schema_with_distractors(example['schema'], example['query'])
    text = (
        "Generate SQL for this question."
        f"### schema\n{schema}\n"
        f"### question\n{example['question']}\n"
        f"### sql\n{example['query']}"
    )
    return text

def extract_schema_with_distractors(schema: str, query: str, n_distractors: int = 5) -> str:
    query_upper = query.upper()
    table_blocks = [b.strip() for b in re.split(r'(?=CREATE TABLE)', schema) if b.strip()]
    
    relevant = []
    distractors = []
    for block in table_blocks:
        match = re.search(r'CREATE TABLE (\w+)', block)
        if match and match.group(1).upper() in query_upper:
            relevant.append(block)
        else:
            distractors.append(block)
    
    selected_distractors = random.sample(distractors, min(n_distractors, len(distractors)))
    combined = relevant + selected_distractors
    random.shuffle(combined)
    return ';\n'.join(combined)