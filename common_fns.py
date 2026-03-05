
def formatting_prompts_func(example: dict):
    text = (
        "Generate SQL for this question."
        f"### schema\n{example['schema']}\n"
        f"### question\n{example['question']}\n"
        f"### sql\n{example['query']}"
    )
    return text