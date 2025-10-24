"""
Inference utilities for fine-tuned models
Handles loading and generating responses from PEFT-trained models
"""

import os
import logging
import torch
import gc
import psutil
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

# Configure logging to show on console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def load_merged_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a merged model (LoRA weights already merged with base model).
    This is faster and simpler than loading PEFT models.

    Args:
        model_path: Path to the merged model directory

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        # Convert relative path to absolute path
        model_path = os.path.abspath(model_path)
        logger.info(f"Loading merged model from: {model_path}")

        # Determine best device
        if torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float32
            logger.info("Using MPS (Metal Performance Shaders) for inference with float32")
        elif torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            logger.info("Using CUDA for inference")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            logger.info("Using CPU for inference")

        # Load merged model directly (no PEFT needed)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True
        ).to(device)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        logger.info("Merged model loaded successfully (no PEFT required)")

        # Store device info for later use
        model.device_info = device

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading merged model: {e}")
        raise


def load_model(model_path: str, base_model: str = "gpt2") -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load a fine-tuned model with LoRA adapters.

    Args:
        model_path: Path to the fine-tuned model directory
        base_model: Base model name (default: gpt2)

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        # Convert relative path to absolute path
        model_path = os.path.abspath(model_path)
        logger.info(f"Loading base model: {base_model}")

        # Determine best device
        if torch.backends.mps.is_available():
            device = "mps"
            device_map = None  # MPS doesn't support device_map
            torch_dtype = torch.float32  # MPS doesn't support float16 for LayerNorm
            logger.info("Using MPS (Metal Performance Shaders) for inference with float32")
        elif torch.cuda.is_available():
            device = "cuda"
            device_map = "auto"
            torch_dtype = torch.float16
            logger.info("Using CUDA for inference")
        else:
            device = "cpu"
            device_map = "cpu"
            torch_dtype = torch.float32
            logger.info("Using CPU for inference")

        # Load base model
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            use_cache=False,  # Disable cache to reduce memory
            low_cpu_mem_usage=True  # Enable memory optimization
        )

        # Move to device if using MPS
        if device == "mps":
            base_model_obj = base_model_obj.to(device)


        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Check available memory before loading PEFT
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"Available memory: {available_memory_gb:.2f} GB")

        if available_memory_gb < 2.0:
            logger.warning("Low memory detected. Consider using a smaller model.")

        logger.info(f"Loading PEFT adapters from: {model_path}")
        model = PeftModel.from_pretrained(base_model_obj, model_path)

        # Skip merging - keep as PEFT model to avoid corruption
        logger.info("Keeping model as PEFT (not merging) for stability")

        # Force garbage collection
        gc.collect()

        model.eval()
        logger.info("Model loaded successfully")

        # Store device info for later use
        model.device_info = device

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def generate_response(model, tokenizer: AutoTokenizer, prompt: str,
                     max_new_tokens: int = 200, temperature: float = 0.7,
                     do_sample: bool = True) -> str:
    """
    Generate a response from the fine-tuned model.

    Args:
        model: The loaded model (can be PEFT or merged)
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling

    Returns:
        Generated response text
    """
    try:
        # Format prompt for instruction following
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize input
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Move inputs to the same device as model
        device = getattr(model, 'device_info', 'cpu')
        if device != 'cpu':
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response with memory protection
        try:
            # Force garbage collection before generation
            gc.collect()

            # Ensure tensors are properly aligned
            input_ids = inputs["input_ids"].contiguous()
            attention_mask = inputs["attention_mask"].contiguous()

            with torch.no_grad():
                # Generate with configurable parameters
                generation_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'max_new_tokens': max_new_tokens,
                    'do_sample': do_sample,
                    'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'use_cache': False,
                    'output_attentions': False,
                    'output_hidden_states': False,
                    'return_dict_in_generate': False
                }

                # Add temperature if using sampling
                if do_sample:
                    generation_kwargs['temperature'] = temperature
                    generation_kwargs['top_p'] = 0.9  # Nucleus sampling
                else:
                    # Use greedy decoding
                    generation_kwargs['num_beams'] = 1

                outputs = model.generate(**generation_kwargs)

        except RuntimeError as e:
            logger.error(f"Generation failed with RuntimeError: {e}")
            return "Error: Model generation failed due to memory constraints"
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            return f"Error: {str(e)}"

        # Decode response
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        response = generated[len(formatted_prompt):].strip()

        # Clean up response
        if "<|im_end|" in response:
            response = response.split("<|im_end|")[0].strip()

        return response

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"


def test_inference(model_path: str = "./models/finetuned_model",
                  base_model: str = "gpt2"):
    """Test the inference system with a simple query."""
    try:
        print("Loading model...")
        model, tokenizer = load_model(model_path, base_model)

        print("Testing inference...")
        test_query = "What is the incorporation date of ALTAI CASHMERE LLC?"
        response = generate_response(model, tokenizer, test_query)

        print(f"\nQuery: {test_query}")
        print(f"Response: {response}")
        print("\nInference test completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_inference()