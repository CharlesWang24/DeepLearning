from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str="homework/rft_model",
    **kwargs,
):
    
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    
    from .data import Dataset
    from .sft import test_model, TokenizedDataset
    
    def rft_format_example(prompt: str, answer: float, reasoning: str) -> dict[str, str]:
        """
        Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
        """
        return {
            "question": prompt,
            "answer": reasoning,
        }
    
    llm = BaseLLM()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(llm.model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    train_data = Dataset("rft")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_data, rft_format_example)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=5,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    test_model(output_dir)
    


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
