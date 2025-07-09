#!/usr/bin/env python3
"""
Simple LoRA training script for basic Q&A
Tests the training pipeline with minimal data
Now includes GGUF export functionality
"""
import os
import subprocess
import sys
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
import json
from huggingface_hub import login
from dotenv import load_dotenv


def authenticate_huggingface():
    """Authenticate with HuggingFace using token from .env file"""
    try:
        # Load environment variables from .env file
        load_dotenv()
        hf_token = os.getenv("outlines_core")

        if hf_token is None:
            print("Warning: HuggingFace token not found in .env file")
            print("Please add 'outlines_core=your_token_here' to .env file")
            return False

        # Login to HuggingFace
        login(token=hf_token)
        print("✓ Successfully authenticated with HuggingFace")
        return True

    except Exception as e:
        print(f"Failed to authenticate with HuggingFace: {e}")
        return False


def load_simple_training_data(file_path="lora_training/econ_fewshots.json"):
    """Load the simple training data"""
    if not os.path.exists(file_path):
        print(f"Training data not found: {file_path}")
        print("Please run: julia generate_simple_test_data.jl")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} simple training examples")
    return data


def format_training_text(examples):
    """Format economics examples for training"""
    formatted_texts = []

    for item in examples:
        # Convert economics problem structure to training format
        instruction = f"Solve this {item['problem_type']} problem:\n\n{item['problem_statement']}"

        # Format the solution steps and final answer
        response = "Solution:\n"

        # Add solution steps if they exist
        if item.get('solution_steps') and len(item['solution_steps']) > 0:
            for i, step in enumerate(item['solution_steps'], 1):
                if step.strip():  # Only add non-empty steps
                    response += f"{i}. {step}\n"

        # Add final answer if it exists
        if item.get('final_answer') and item['final_answer'].strip():
            response += f"\nFinal Answer: {item['final_answer']}"
        else:
            response += f"\nFinal Answer: [Answer needs to be provided]"

        # Create training format
        text = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
        formatted_texts.append(text)

    return formatted_texts


def setup_model_and_tokenizer(model_name="google/gemma-2-2b-it"):
    """Setup model and tokenizer with HuggingFace authentication"""
    print(f"Loading model: {model_name}")

    # Authenticate first
    if not authenticate_huggingface():
        print("Warning: Proceeding without authentication - may fail for gated models")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=True,  # Updated parameter name
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=True,  # Updated parameter name
            trust_remote_code=True
        )

        print(f"✓ Successfully loaded {model_name}")
        return model, tokenizer

    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        print("This might be due to:")
        print("1. Missing HuggingFace token")
        print("2. No access to Gemma model")
        print("3. Network issues")
        raise e


def setup_simple_lora_config():
    """LoRA configuration for Gemma-2-2b"""
    lora_config = LoraConfig(
        r=16,  # Rank for Gemma
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj"       # MLP layers
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    return lora_config


def tokenize_data(texts, tokenizer, max_length=512):
    """Tokenize the training texts"""
    def tokenize_function(examples):
        # Tokenize each text individually with proper padding and truncation
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,  # Add padding
            max_length=max_length,
            return_tensors=None  # Don't return tensors yet
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Create dataset
    dataset = Dataset.from_dict({"text": texts})

    # Tokenize with batched=False to handle variable lengths better
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    return tokenized_dataset


def check_llama_cpp_availability():
    """Check if llama.cpp is available for GGUF conversion"""
    # Check vendors/llama.cpp directory
    convert_script = Path("vendors/llama.cpp/convert_lora_to_gguf.py")

    if convert_script.exists():
        print(f"✓ Found llama.cpp at: vendors/llama.cpp")
        return str(convert_script)
    else:
        print(f"❌ convert_lora_to_gguf.py not found at: {convert_script}")
        print("Please ensure llama.cpp is properly set up in vendors/llama.cpp/")
        return None


def convert_lora_to_gguf(adapter_path, output_name="economics_lora.gguf"):
    """Convert the trained LoRA adapter to GGUF format"""
    print("\n" + "="*50)
    print("CONVERTING LORA TO GGUF FORMAT")
    print("="*50)

    # Check for llama.cpp in vendors directory
    convert_script = check_llama_cpp_availability()
    if convert_script is None:
        print("❌ Cannot convert to GGUF: convert_lora_to_gguf.py not found")
        print("Please ensure llama.cpp is properly set up in vendors/llama.cpp/")
        return False

    # Prepare conversion command
    cmd = [
        sys.executable,
        convert_script,
        adapter_path,
        "--outfile", output_name,
        "--outtype", "f16"  # Use f16 for good balance of size/quality
    ]

    print(f"🔄 Converting LoRA adapter to GGUF...")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Run conversion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd()
        )

        # Check if output file was created
        if os.path.exists(output_name):
            file_size = os.path.getsize(output_name) / (1024 * 1024)  # MB
            print(f"✅ Successfully converted to GGUF!")
            print(f"   Output file: {output_name}")
            print(f"   File size: {file_size:.1f} MB")
            print(f"   This file can now be used with your Julia llama.cpp bindings")
            return True
        else:
            print(
                f"❌ Conversion completed but output file not found: {output_name}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed!")
        print(f"Error output: {e.stderr}")
        print(f"Command output: {e.stdout}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during conversion: {e}")
        return False


def create_usage_instructions(gguf_path, adapter_name="economics_lora"):
    """Create usage instructions for the GGUF LoRA"""
    instructions = f"""
# LoRA GGUF Usage Instructions

Your economics LoRA has been successfully converted to GGUF format!

## File Information:
- GGUF file: {gguf_path}
- Adapter name: {adapter_name}
- Compatible with: llama.cpp and your Julia bindings

## Julia Usage Example:

```julia
# Load your trained LoRA adapter
economics_adapter = load_lora_adapter(model, "{gguf_path}")

# Apply the adapter to your context
result = set_adapter_lora(ctx, economics_adapter, 1.0f0)  # Full strength
if result == 0
    println("Economics LoRA applied successfully")
end

# Generate text with economics specialization
response = generate_text("Solve this economics problem: ...", 100)

# Remove adapter when done
rm_adapter_lora(ctx, economics_adapter)

# Or switch to different strength
set_adapter_lora(ctx, economics_adapter, 0.5f0)  # Half strength
```

## LoRA Manager Usage:

```julia
# Using the LoRA manager
lora_manager = LoRAManager(model)
load_adapter!(lora_manager, "{adapter_name}", "{gguf_path}")
switch_adapter!(lora_manager, ctx, "{adapter_name}", scale=1.0f0)
```

## SEAL Integration:

This LoRA can now be integrated into your SEAL economics pipeline for
dynamic model specialization and self-improvement.
"""

    with open("gguf_usage_instructions.md", "w", encoding='utf-8') as f:
        f.write(instructions)

    print(f"Usage instructions saved to: gguf_usage_instructions.md")


def train_simple_lora():
    """Main training function - simple and fast"""
    print("Gemma-2-2B LoRA Training")
    print("=" * 40)

    # 1. Load data
    training_data = load_simple_training_data()
    if training_data is None:
        return None, None

    # 2. Setup model and tokenizer (always use Gemma)
    model, tokenizer = setup_model_and_tokenizer("google/gemma-2-2b-it")

    # 3. Format data
    formatted_texts = format_training_text(training_data)

    # 4. Tokenize
    tokenized_dataset = tokenize_data(formatted_texts, tokenizer)

    # 5. Setup LoRA
    lora_config = setup_simple_lora_config()
    model = get_peft_model(model, lora_config)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / all_params:.2f}%")

    # 6. Training arguments - more conservative for small dataset
    training_args = TrainingArguments(
        output_dir="./gemma_lora_test",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Reduce epochs to prevent overfitting
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,  # Reduce accumulation
        warmup_steps=5,  # Reduce warmup
        learning_rate=5e-5,  # Lower learning rate
        fp16=torch.cuda.is_available(),
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None,
        max_grad_norm=1.0,  # Add gradient clipping
        weight_decay=0.01,  # Add regularization
    )

    # 7. Data collator with padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Pad to multiples of 8 for efficiency
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 9. Train
    print("Starting training...")
    trainer.train()

    # 10. Save HuggingFace format first
    adapter_dir = "./gemma_lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("Training completed!")
    print(f"Adapter saved to: {adapter_dir}")

    # 11. Convert to GGUF format
    gguf_filename = "economics_lora_gemma2_2b.gguf"
    conversion_success = convert_lora_to_gguf(adapter_dir, gguf_filename)

    if conversion_success:
        # Create usage instructions
        create_usage_instructions(gguf_filename, "economics_gemma2")

        print(f"\n🎉 LoRA training and GGUF conversion completed!")
        print(f"   HuggingFace format: {adapter_dir}")
        print(f"   GGUF format: {gguf_filename}")
        print(f"   Ready for Julia llama.cpp bindings!")
    else:
        print(f"\n⚠️  LoRA training completed but GGUF conversion failed")
        print(f"   HuggingFace format available at: {adapter_dir}")
        print(f"   You can manually convert later using llama.cpp")

    return model, tokenizer


def test_simple_model():
    """Test the trained model"""
    print("\nTesting trained Gemma model...")

    # Load base model with same settings
    base_model, tokenizer = setup_model_and_tokenizer("google/gemma-2-2b-it")

    # Load LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, "./gemma_lora_adapter")

    # Set model to eval mode
    model.eval()

    # Simple test prompts
    test_prompts = [
        "<bos><start_of_turn>user\nWhat is 2 + 2?<end_of_turn>\n<start_of_turn>model\n",
        "<bos><start_of_turn>user\nSolve this problem: A farmer has 10 acres of land.<end_of_turn>\n<start_of_turn>model\n"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(
            f"Input: {prompt.split('?')[0] if '?' in prompt else prompt.split('.')[0]}...")

        try:
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=256)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,  # Use greedy decoding to avoid sampling issues
                    temperature=None,  # Disable temperature when using greedy
                    top_p=None,       # Disable top_p when using greedy
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=True
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = response[len(tokenizer.decode(
                inputs['input_ids'][0], skip_special_tokens=True)):].strip()

            print(f"Generated: {generated_part}")

        except Exception as e:
            print(f"Generation failed: {e}")
            print("This might be due to:")
            print("- Model overfitting (loss went to 0)")
            print("- Numerical instability")
            print("- Try training with lower learning rate or fewer epochs")


if __name__ == "__main__":
    try:
        # Check for required packages
        try:
            from dotenv import load_dotenv
        except ImportError:
            print("Installing python-dotenv...")
            os.system("pip install python-dotenv")
            from dotenv import load_dotenv

        # Train
        model, tokenizer = train_simple_lora()

        if model is not None:
            # Test
            test_simple_model()
            print("\n✅ Gemma LoRA training and GGUF export completed successfully!")
            print("Your economics LoRA is ready for use with Julia llama.cpp bindings!")
        else:
            print("❌ Training failed")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
