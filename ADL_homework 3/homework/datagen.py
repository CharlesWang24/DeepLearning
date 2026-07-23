def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6, model_checkpoint: str | None=None):
    import json
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid
    
    model = CoTModel(checkpoint=model_checkpoint) if model_checkpoint else CoTModel()
    trainset = Dataset("train")
    questions = [trainset[i][0] for i in range(len(trainset))]
    correct_answers = [trainset[i][1] for i in range(len(trainset))]
    
    prompts = [model.format_prompt(q) for q in questions]
    generations = model.batched_generate(prompts, temperature=temperature, num_return_sequences=oversample)
    
    dataset = []
    
    for question, correct, generated in zip(questions, correct_answers, generations):
        for g in generated:
            answer = model.parse_answer(g)
            if is_answer_valid(answer, correct):
                dataset.append([question, answer, correct])
                break
            
    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
