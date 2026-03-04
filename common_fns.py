
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = (
            f"<schema>\n{example['context'][i]}\n"
            f"<question>\n{example['question'][i]}\n"
            f"<query>\n{example['answer'][i]}"
        )
        output_texts.append(text)
    return output_texts