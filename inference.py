#!/usr/bin/env python3
"""
Inference script for fine-tuned QLoRA model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_path="./models/finetuned_model", base_model="Qwen/Qwen2.5-1.5B-Instruct"):
    """Load the fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_new_tokens=200, temperature=0.5):
    """Generate response for an instruction."""
    input_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
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
    
    test_instruction = "What is the name of the company with company ID L61305031N?"
    response = generate_response(model, tokenizer, test_instruction)
    print(f"Instruction: {test_instruction}")
    print(f"Response: {response}")
