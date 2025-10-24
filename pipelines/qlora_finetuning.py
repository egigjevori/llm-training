"""
Simplified QLoRA Fine-tuning Pipeline using ZenML
MacBook-optimized with TinyLlama for fast training
"""

import logging
import json
import os
from typing import Dict
from dataclasses import dataclass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenml import pipeline, step
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleConfig:
    """Simplified configuration for QLoRA fine-tuning."""
    model_name: str = "gpt2"
    dataset_path: str = "./datasets/instruction_dataset.jsonl"
    output_dir: str = "models/finetuned_model"
    learning_rate: float = 3e-4  # Slightly higher learning rate
    num_epochs: int = 3  # More epochs for better learning
    batch_size: int = 4
    lora_r: int = 16  # Higher rank for more parameters
    lora_alpha: int = 32  # Higher alpha for more adaptation
    

@step(enable_cache=False)
def load_and_prepare_data(config: SimpleConfig) -> Dataset:
    """Load and format instruction dataset."""
    logger.info(f"Loading dataset from {config.dataset_path}")

    # Load JSONL data
    data = []
    with open(config.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Format for instruction tuning
    def format_text(example):
        return {"text": f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"}

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_text, remove_columns=dataset.column_names)

    logger.info(f"Prepared {len(dataset)} training samples")
    return dataset

@step(enable_cache=False)
def setup_and_train(dataset: Dataset, config: SimpleConfig) -> str:
    """Setup model, tokenizer and train with QLoRA."""
    logger.info(f"Loading model: {config.model_name}")

    import torch
    import platform

    # Determine best device for MacBook
    if torch.backends.mps.is_available():
        device = "mps"
        device_map = None  # MPS doesn't support device_map
        torch_dtype = torch.float32  # MPS doesn't support float16 for LayerNorm
        logger.info("Using MPS (Metal Performance Shaders) for acceleration with float32")
    elif torch.cuda.is_available():
        device = "cuda"
        device_map = "auto"
        torch_dtype = torch.float16
        logger.info("Using CUDA for acceleration")
    else:
        device = "cpu"
        device_map = "cpu"
        torch_dtype = torch.float32
        logger.info("Using CPU (no acceleration available)")

    logger.info(f"Loading model on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Move to device if using MPS
    if device == "mps":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        bias="none",
        target_modules=["c_attn", "c_proj"],
    )

    # Training arguments with device-specific optimizations
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_steps=50,  # Save more frequently
        eval_strategy="no",
        warmup_steps=50,  # More warmup steps
        weight_decay=0.01,  # Add weight decay for regularization
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to="none",
        fp16=device == "cuda",  # Only enable fp16 for CUDA (not MPS due to Accelerate limitation)
        dataloader_pin_memory=False if device == "mps" else True,  # MPS doesn't support pin_memory
        use_cpu=device == "cpu",  # Use CPU when device is CPU
        gradient_accumulation_steps=2,  # Effectively double batch size
        max_grad_norm=1.0,  # Gradient clipping for stability
    )

    # Create trainer and train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    logger.info(f"Model saved to {config.output_dir}")
    return config.output_dir

@step(enable_cache=False)
def evaluate(model_path: str, config: SimpleConfig) -> Dict:
    """Simple evaluation with test examples."""
    from peft import PeftModel
    import torch

    logger.info("Loading model for evaluation")

    # Determine device for evaluation
    if torch.backends.mps.is_available():
        device = "mps"
        device_map = None
        torch_dtype = torch.float32  # MPS doesn't support float16 for LayerNorm
        logger.info("Using MPS for evaluation with float32")
    elif torch.cuda.is_available():
        device = "cuda"
        device_map = "auto"
        torch_dtype = torch.float16
        logger.info("Using CUDA for evaluation")
    else:
        device = "cpu"
        device_map = "cpu"
        torch_dtype = torch.float32
        logger.info("Using CPU for evaluation")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Move to device if using MPS
    if device == "mps":
        base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # Test examples relevant to your dataset
    test_cases = [
        "What is the company number for ! LTD?",
        "What is the incorporation date of ALTAI CASHMERE LLC?",
        "What type of company is a Private Limited Company?",
        "In which town is a company with postcode HG1 1ND likely registered?",
        "What does company status 'Active' mean?"
    ]

    results = []
    for instruction in test_cases:
        input_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

        # Move inputs to device
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(input_text):].strip()

        results.append({
            "instruction": instruction,
            "response": response
        })

        logger.info(f"Q: {instruction}")
        logger.info(f"A: {response}\n")

    # Save results
    eval_results = {"test_results": results, "model_path": model_path}
    with open(os.path.join(model_path, "evaluation_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    return eval_results


@pipeline
def qlora_finetuning_pipeline(
    dataset_path: str = "../datasets/instruction_dataset.jsonl",
    model_name: str = "gpt2",
    num_epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    output_dir: str = "../models/finetuned_model"
):
    """Simplified QLoRA fine-tuning pipeline."""

    # Create simple configuration
    config = SimpleConfig(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    # Step 1: Load and prepare data
    dataset = load_and_prepare_data(config)

    # Step 2: Setup and train model
    model_path = setup_and_train(dataset, config)

    logger.info("QLoRA fine-tuning completed!")
    logger.info(f"Model saved to: {model_path}")

    return {
        "model_path": model_path
    }


if __name__ == "__main__":
    qlora_finetuning_pipeline()
