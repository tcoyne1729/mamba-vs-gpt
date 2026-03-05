
def formatting_prompts_func(example: dict):
    print(list(example.keys()))
    output_texts = []
    for i in range(len(example['question'])):
        text = (
            "Generate SQL for this question."
            f"### schema\n{example['schema'][i]}\n"
            f"### question\n{example['question'][i]}\n"
            f"### sql\n{example['query'][i]}"
        )
        output_texts.append(text)
    return output_texts