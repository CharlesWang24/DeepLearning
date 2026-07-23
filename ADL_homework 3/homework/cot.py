from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {
                "role": "system",
                "content": ("You are a precise unit conversion assistant. State the conversion factor and the arithmetic in one short line, then give the final numeric result inside <answer></answer> tags. Be concise.")
            },
            {
                "role": "user",
                "content": "How many grams are there per 6kg?"
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"
            },
            {
                "role": "user",
                "content": "How many meters are there per 3km?"
            },
            {
                "role": "assistant",
                "content": "1 km = 1000 meters. 3 * 1000 = <answer>3000</answer>"
            },
            {
                "role": "user",
                "content": "How many minutes are there per 2 hours?"
            },
            {
                "role": "assistant",
                "content": "1 hour = 60 minutes. 2 * 60 = <answer>120</answer>"
            },
            {
                "role": "user",
                "content": question
            },
            
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
