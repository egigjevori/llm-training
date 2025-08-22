"""
QLoRA Fine-tuning Pipeline for LLMs using ZenML
Quantized Low-Rank Adaptation for efficient fine-tuning
"""

import logging
import json
import os
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenml import pipeline, step
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for QLoRA fine-tuning."""
    # Model configuration
    base_model: str = "microsoft/DialoGPT-medium"  # Can be changed to other models
    max_length: int = 512
    padding: bool = True
    truncation: bool = True
    
    # QLoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Will auto-detect if None
    
    # Training configuration
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Quantization configuration
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Output configuration
    output_dir: str = "models/finetuned_model"
    save_total_limit: int = 3
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for most transformer models
            # For DialoGPT, we'll use the attention modules
            self.target_modules = ["c_attn", "c_proj", "c_fc", "c_proj"]


@step
def load_instruction_dataset(dataset_path: str = "datasets/instruction_dataset.jsonl") -> Dataset:
    """Load and prepare the instruction dataset."""
    try:
        logger.info(f"Loading instruction dataset from {dataset_path}")
        
        # Read JSONL file
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(data)} instruction-response pairs")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)
        
        logger.info(f"Dataset created with {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


@step
def prepare_model_and_tokenizer(config: TrainingConfig) -> Tuple[object, object]:
    """Prepare the model and tokenizer for QLoRA fine-tuning."""
    try:
        logger.info(f"Loading base model: {config.base_model}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check if we can use 4-bit quantization (not available on macOS ARM64)
        try:
            import bitsandbytes
            use_4bit = config.load_in_4bit
            logger.info("bitsandbytes available, using 4-bit quantization")
        except ImportError:
            use_4bit = False
            logger.warning("bitsandbytes not available, falling back to full precision")
        
        if use_4bit:
            # Load model with quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit,
                bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
        else:
            # Load model without quantization
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Auto-detect target modules if not specified
        if config.target_modules is None:
            # Get all module names
            module_names = set()
            for name, _ in model.named_modules():
                module_names.add(name.split('.')[-1])
            
            # Common target modules for different model architectures
            common_targets = {
                "c_attn", "c_proj", "c_fc",  # GPT-2/DialoGPT style
                "q_proj", "v_proj", "k_proj", "o_proj",  # LLaMA style
                "gate_proj", "up_proj", "down_proj",  # LLaMA MLP
                "query", "key", "value", "dense",  # BERT style
                "fc1", "fc2", "proj"  # Generic
            }
            
            # Find matching modules
            target_modules = list(common_targets.intersection(module_names))
            if not target_modules:
                # Fallback to common attention modules
                target_modules = ["c_attn", "c_proj"]
            
            logger.info(f"Auto-detected target modules: {target_modules}")
        else:
            target_modules = config.target_modules
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        logger.info("Model and tokenizer prepared successfully")
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Error preparing model and tokenizer: {e}")
        raise


@step
def preprocess_dataset(dataset: Dataset, tokenizer: object, config: TrainingConfig) -> Dataset:
    """Preprocess the dataset for training."""
    try:
        logger.info("Preprocessing dataset for training")
        
        def format_instruction_response(example):
            """Format instruction-response pairs for training."""
            instruction = example.get('instruction', '')
            response = example.get('response', '')
            
            # Format as instruction-following prompt
            formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n\n### End\n"
            return {"text": formatted_text}
        
        def tokenize_function(examples):
            """Tokenize the text data."""
            return tokenizer(
                examples["text"],
                truncation=config.truncation,
                padding=config.padding,
                max_length=config.max_length,
                return_tensors="pt"
            )
        
        # Format the dataset
        formatted_dataset = dataset.map(format_instruction_response, remove_columns=dataset.column_names)
        
        # Tokenize the dataset
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names
        )
        
        logger.info(f"Dataset preprocessed. Final size: {len(tokenized_dataset)}")
        return tokenized_dataset
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        raise


@step
def setup_training_arguments(config: TrainingConfig) -> TrainingArguments:
    """Setup training arguments for the trainer."""
    try:
        logger.info("Setting up training arguments")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Check if we're on MPS (Apple Silicon) and disable fp16
        import torch
        use_fp16 = True
        if torch.backends.mps.is_available():
            use_fp16 = False
            logger.info("MPS detected, disabling fp16 mixed precision")
        
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            fp16=use_fp16,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard logging
        )
        
        logger.info("Training arguments configured successfully")
        return training_args
        
    except Exception as e:
        logger.error(f"Error setting up training arguments: {e}")
        raise


@step
def train_model(
    model: object,
    tokenizer: object,
    dataset: Dataset,
    training_args: TrainingArguments,
    config: TrainingConfig
) -> str:
    """Train the model using QLoRA."""
    try:
        logger.info("Starting QLoRA fine-tuning")
        
        # Split dataset into train and validation
        dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(eval_dataset)}")
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        logger.info("Saving fine-tuned model...")
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        
        # Save training config
        config_dict = {
            "base_model": config.base_model,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": config.target_modules,
            "training_args": training_args.to_dict(),
        }
        
        with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {config.output_dir}")
        return config.output_dir
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


@step
def evaluate_model(
    model_path: str,
    tokenizer: object,
    test_dataset: Dataset,
    config: TrainingConfig
) -> Dict:
    """Evaluate the fine-tuned model."""
    try:
        logger.info("Evaluating fine-tuned model")
        
        # Load the fine-tuned model
        from peft import PeftModel
        
        # Check if we can use 4-bit quantization
        try:
            import bitsandbytes
            use_4bit = True
            logger.info("bitsandbytes available for evaluation")
        except ImportError:
            use_4bit = False
            logger.warning("bitsandbytes not available for evaluation, using full precision")
        
        if use_4bit:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Try MPS first, fallback to CPU if issues
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.base_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"MPS loading failed, falling back to CPU: {e}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.base_model,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        # Test on a few examples
        test_results = []
        test_samples = test_dataset.select(range(min(5, len(test_dataset))))
        
        for i, sample in enumerate(test_samples):
            instruction = sample.get('instruction', '')
            expected_response = sample.get('response', '')
            
            # Format input
            input_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config.max_length)
            
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(input_text):].strip()
            
            test_results.append({
                "instruction": instruction,
                "expected": expected_response,
                "generated": response,
                "sample_id": i
            })
        
        # Calculate basic metrics
        evaluation_metrics = {
            "test_samples": len(test_results),
            "model_path": model_path,
            "test_results": test_results
        }
        
        # Save evaluation results
        eval_path = os.path.join(model_path, "evaluation_results.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation completed. Results saved to {eval_path}")
        return evaluation_metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


@step
def create_inference_pipeline(model_path: str, config: TrainingConfig) -> str:
    """Create an inference pipeline for the fine-tuned model."""
    try:
        logger.info("Creating inference pipeline")
        
        inference_script = f'''#!/usr/bin/env python3
"""
Inference script for fine-tuned QLoRA model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_path="{model_path}", base_model="{config.base_model}"):
    """Load the fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_new_tokens=200, temperature=0.7):
    """Generate response for an instruction."""
    input_text = f"### Instruction:\\n{{instruction}}\\n\\n### Response:\\n"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(input_text):].strip()
    
    return response

if __name__ == "__main__":
    # Example usage
    model, tokenizer = load_model()
    
    test_instruction = "What is the capital of Albania?"
    response = generate_response(model, tokenizer, test_instruction)
    print(f"Instruction: {{test_instruction}}")
    print(f"Response: {{response}}")
'''
        
        # Save inference script
        inference_path = os.path.join(model_path, "inference.py")
        with open(inference_path, "w") as f:
            f.write(inference_script)
        
        # Make it executable
        os.chmod(inference_path, 0o755)
        
        logger.info(f"Inference pipeline created at {inference_path}")
        return inference_path
        
    except Exception as e:
        logger.error(f"Error creating inference pipeline: {e}")
        raise


@pipeline
def qlora_finetuning_pipeline(
    dataset_path: str = "datasets/instruction_dataset.jsonl",
    base_model: str = "microsoft/DialoGPT-medium",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    output_dir: str = "models/finetuned_model"
):
    """Main QLoRA fine-tuning pipeline."""
    
    # Create training configuration
    config = TrainingConfig(
        base_model=base_model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir
    )
    
    # Load and prepare dataset
    dataset = load_instruction_dataset(dataset_path)
    
    # Prepare model and tokenizer
    tokenizer, model = prepare_model_and_tokenizer(config)
    
    # Preprocess dataset
    processed_dataset = preprocess_dataset(dataset, tokenizer, config)
    
    # Setup training arguments
    training_args = setup_training_arguments(config)
    
    # Train the model
    model_path = train_model(model, tokenizer, processed_dataset, training_args, config)
    
    # Evaluate the model
    evaluation_results = evaluate_model(model_path, tokenizer, processed_dataset, config)
    
    # Create inference pipeline
    inference_path = create_inference_pipeline(model_path, config)
    
    logger.info("QLoRA fine-tuning pipeline completed successfully!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Inference script: {inference_path}")
    
    return {
        "model_path": model_path,
        "inference_path": inference_path,
        "evaluation_results": evaluation_results
    }


if __name__ == "__main__":
    qlora_finetuning_pipeline() 