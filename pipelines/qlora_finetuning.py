"""
QLoRA Fine-tuning Pipeline for LLMs using ZenML
Quantized Low-Rank Adaptation for efficient fine-tuning
"""

import logging
import json
import os
import torch
import math
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field

from datasets.arrow_dataset import Dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenml import pipeline, step
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumericalStabilityCallback(TrainerCallback):
    """Callback to monitor numerical stability during training."""
    
    def __init__(self):
        self.step_count = 0
        self.nan_detected = False
        
    def on_step_end(self, args, state, control, **kwargs):
        """Check for numerical issues after each step."""
        self.step_count += 1
        
        # Get the model from kwargs
        model = kwargs.get('model')
        if model is None:
            return control
            
        # Check for NaN in model parameters
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    logger.error(f"Step {self.step_count}: NaN detected in gradients for {name}")
                    self.nan_detected = True
                    control.should_training_stop = True
                    return control
                    
                # Check gradient norm
                grad_norm = param.grad.data.norm()
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.error(f"Step {self.step_count}: Invalid gradient norm for {name}: {grad_norm}")
                    self.nan_detected = True
                    control.should_training_stop = True
                    return control
                    
                # Log large gradients
                if grad_norm > 100.0:
                    logger.warning(f"Step {self.step_count}: Large gradient norm for {name}: {grad_norm:.4f}")
                    
            # Check for NaN in parameters themselves
            if torch.isnan(param.data).any():
                logger.error(f"Step {self.step_count}: NaN detected in parameter {name}")
                self.nan_detected = True
                control.should_training_stop = True
                return control
                
        return control
        
    def on_log(self, args, state, control, **kwargs):
        """Monitor logged metrics for issues."""
        logs = kwargs.get('logs', {})
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    logger.error(f"Step {self.step_count}: Invalid value in logs - {key}: {value}")
                    self.nan_detected = True
                    control.should_training_stop = True
                    return control
                    
                # Log unusual loss values
                if key == 'loss':
                    if value == 0.0:
                        logger.warning(f"Step {self.step_count}: Loss is exactly 0.0 - possible numerical issue")
                    elif value > 1000:
                        logger.warning(f"Step {self.step_count}: Very high loss: {value}")
                        
        return control

@dataclass
class TrainingConfig:
    """Configuration for QLoRA fine-tuning."""
    # Model configuration
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Smaller model for faster training
    max_length: int = 512
    padding: bool = True
    truncation: bool = True
    
    # QLoRA configuration
    lora_r: int = 16  # Increased for more adaptation capacity
    lora_alpha: int = 32  # Increased for stronger adaptation
    lora_dropout: float = 0.05  # Reduced for less regularization
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Training configuration
    num_epochs: int = 3
    batch_size: int = 4  # Higher batch size for smaller model
    gradient_accumulation_steps: int = 2  # Reduced steps (effective batch = 8) 
    learning_rate: float = 1e-5  # Slightly higher LR for smaller model
    warmup_steps: int = 50  # Gentle warmup
    weight_decay: float = 0.001  # Reduced regularization
    logging_steps: int = 10
    save_steps: int = 100  # More frequent saves
    eval_steps: int = 100  # More frequent evaluation

    # Memory optimization configuration
    gradient_checkpointing: bool = True  # Enable gradient checkpointing
    use_flash_attention: bool = True  # Use flash attention if available
    max_memory: Dict[str, str] = field(default_factory=lambda: {"mps": "8GB", "cpu": "16GB"})
    offload_folder: str = "offload"  # Folder for CPU offloading
    cpu_offload: bool = True  # Enable CPU offloading for inactive layers
    
    # Output configuration
    output_dir: str = "models/finetuned_model"
    save_total_limit: int = 3
    


@step(enable_cache=False)
def load_instruction_dataset(dataset_path: str = "pipelines/datasets/instruction_dataset.jsonl") -> Dataset:
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


@step(enable_cache=False)
def prepare_model_and_tokenizer(config: TrainingConfig) -> Tuple[PreTrainedTokenizerBase, Union[PreTrainedModel, PeftModel]]:
    """Prepare the model and tokenizer for QLoRA fine-tuning."""
    try:
        logger.info(f"Loading base model: {config.base_model}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create offload directory if CPU offloading is enabled
        if config.cpu_offload:
            os.makedirs(config.offload_folder, exist_ok=True)
            logger.info(f"CPU offload directory created: {config.offload_folder}")
        
        # Load model with float32 for better numerical stability
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float32,  # Use float32 for stability
            device_map="auto",
            trust_remote_code=True,
            max_memory=config.max_memory,
            offload_folder=config.offload_folder if config.cpu_offload else None,
            low_cpu_mem_usage=True,
        )
        
        # Enable gradient checkpointing to save memory
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Configure LoRA with memory-optimized settings
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Print model memory usage
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Memory usage: {total_params * 4 / 1024**3:.2f} GB (estimated)")
        
        logger.info("Model and tokenizer prepared successfully")
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Error preparing model and tokenizer: {e}")
        raise


@step(enable_cache=False)
def preprocess_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, config: TrainingConfig) -> Dataset:
    """Preprocess the dataset for training."""
    try:
        logger.info("Preprocessing dataset for training")
        
        def format_instruction_response(example):
            """Format instruction-response pairs for training."""
            instruction = example.get('instruction', '')
            response = example.get('response', '')
            
            # Format as instruction-following prompt
            formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            return {"text": formatted_text}
        
        def tokenize_function(examples):
            """Tokenize the text data."""
            # Don't return tensors here - let the data collator handle that
            result = tokenizer(
                examples["text"],
                truncation=config.truncation,
                padding=False,  # Let data collator handle padding
                max_length=config.max_length,
                return_tensors=None  # Return lists, not tensors
            )
            # Labels will be created by the data collator
            return result
        
        # Format the dataset with memory-efficient processing
        logger.info("Formatting dataset...")
        formatted_dataset = dataset.map(
            format_instruction_response, 
            remove_columns=dataset.column_names,
            batch_size=100,  # Process in smaller batches to save memory
            desc="Formatting dataset"
        )
        
        # Clear memory after formatting
        import gc
        gc.collect()
        
        # Tokenize the dataset with memory optimization
        logger.info("Tokenizing dataset...")
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=50,  # Smaller batch size for tokenization to save memory
            remove_columns=formatted_dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Clear memory after tokenization
        gc.collect()
        
        logger.info(f"Dataset preprocessed. Final size: {len(tokenized_dataset)}")
        logger.info(f"Memory usage after preprocessing: {tokenized_dataset.data.nbytes / 1024**3:.2f} GB")
        
        return tokenized_dataset
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        raise


@step(enable_cache=False)
def setup_training_arguments(config: TrainingConfig) -> TrainingArguments:
    """Setup training arguments for the trainer."""
    try:
        logger.info("Setting up training arguments")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

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
            fp16=False,  # Disable mixed precision for numerical stability
            bf16=False,  # Also disable bfloat16
            # Learning rate schedule
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,  # 5% of total steps
            # Optimizer settings
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.95,
            # Memory optimization settings
            dataloader_pin_memory=False,  # Disable pin memory to save RAM
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard logging
            # Additional memory optimizations
            dataloader_num_workers=0,  # Reduce worker processes to save memory
            dataloader_prefetch_factor=None,  # Disable prefetching
            # Gradient checkpointing (already enabled in model, but ensure it's set here too)
            gradient_checkpointing=config.gradient_checkpointing,
            # Strict gradient clipping to prevent NaN
            max_grad_norm=0.5,  # More aggressive gradient clipping
            # Save memory during evaluation
            eval_strategy="steps" if config.eval_steps > 0 else "no",
            eval_steps=config.eval_steps,
            # Memory-efficient saving
            save_strategy="steps",
            # Disable unnecessary features to save memory
            load_best_model_at_end=False,  # Don't load best model to save memory
            metric_for_best_model=None,
            greater_is_better=None,
        )
        
        logger.info("Training arguments configured successfully")
        return training_args
        
    except Exception as e:
        logger.error(f"Error setting up training arguments: {e}")
        raise


@step(enable_cache=False)
def train_model(
    model: Union[PreTrainedModel, PeftModel],
    tokenizer: PreTrainedTokenizerBase,
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

        # Setup custom data collator that handles labels properly
        from transformers import default_data_collator
        from transformers.data.data_collator import DataCollatorMixin
        import torch
        
        class CustomDataCollatorForCausalLM:
            def __init__(self, tok):
                self.tokenizer = tok
                
            def __call__(self, batch):
                # Pad the batch
                batch = self.tokenizer.pad(
                    batch,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=8,
                    return_tensors="pt"
                )
                
                # Create labels from input_ids for causal LM
                batch["labels"] = batch["input_ids"].clone()
                
                # Set padding tokens to -100 so they're ignored in loss
                if self.tokenizer.pad_token_id is not None:
                    batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
                    
                return batch
        
        data_collator = CustomDataCollatorForCausalLM(tokenizer)
        
        # Initialize numerical stability callback
        stability_callback = NumericalStabilityCallback()
        
        # Initialize trainer with callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=[stability_callback],
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
            "memory_optimizations": {
                "gradient_checkpointing": config.gradient_checkpointing,
                "use_flash_attention": config.use_flash_attention,
                "cpu_offload": config.cpu_offload,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
            }
        }
        
        with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Model saved to {config.output_dir}")
        return config.output_dir
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


@step(enable_cache=False)
def evaluate_model(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    test_dataset: Dataset,
    config: TrainingConfig
) -> Dict:
    """Evaluate the fine-tuned model."""
    try:
        logger.info("Evaluating fine-tuned model")
        
        # Load the fine-tuned model
        from peft import PeftModel
        
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
        
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to load PeftModel: {e}")
            raise RuntimeError(f"Could not load fine-tuned model from {model_path}. Error: {e}")
        
        # Test on a few examples
        test_results = []
        test_samples = test_dataset.select(range(min(5, len(test_dataset))))
        
        for i in range(len(test_samples)):
            try:
                sample = test_samples[i]
                instruction = sample.get('instruction', '')
                expected_response = sample.get('response', '')
                
                if not instruction:
                    logger.warning(f"Sample {i} has no instruction, skipping")
                    continue
                
                # Format input
                input_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
                
                # Tokenize input
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config.max_length)
                
                # Move inputs to the same device as the model
                try:
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except (StopIteration, RuntimeError):
                    # Fallback to CPU if device detection fails
                    device = torch.device("cpu")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    logger.warning("Device detection failed, using CPU for inputs")
                
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
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
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


@step(enable_cache=False)
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
        torch_dtype=torch.float16,
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
    dataset_path: str = "pipelines/datasets/instruction_dataset.jsonl",
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_epochs: int = 3,
    batch_size: int = 1,
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
