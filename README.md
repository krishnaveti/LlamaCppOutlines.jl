# LlamaCppOutlines.jl

A Julia package for LLaMA inference with structured output generation using Outlines constraints.

## Features

- **LLaMA Model Inference**: Basic text generation with sampling
- **Multimodal Support**: Text + image generation with multimodal models
- **Constrained Generation**: Structured output using JSON schema constraints
- **Enhanced Sampling**: Multiple sampling strategies (greedy, top-k, top-p, temperature)
- **LoRA Support**: Dynamic adapter loading and switching

## Installation

```julia
# Add the package (when published)
] add LlamaCppOutlines

# Or install from local directory
] dev path/to/LlamaCppOutlines
```

## Quick Start

```julia
using LlamaCppOutlines

# Initialize the APIs
init_apis!()

# Load a model
model, model_context, vocab = load_and_initialize("path/to/model.gguf")

# Basic text generation
result = generate_with_sampling(
    "What is the capital of France?",
    model=model,
    model_context=model_context,
    vocab=vocab,
    max_new_tokens=50
)

# Constrained generation with JSON schema
schema = Dict(
    "type" => "object",
    "properties" => Dict(
        "city" => Dict("type" => "string"),
        "country" => Dict("type" => "string")
    ),
    "required" => ["city", "country"]
)

result = greedy_constrained_generation(
    "What is the capital of France?",
    schema,
    "google/gemma-2-2b-it",
    model_context=model_context,
    vocab=vocab
)
```

## Multimodal Usage

```julia
# Load multimodal model
model, model_context, mtmd_context, vocab = load_and_initialize_mtmd(
    "path/to/model.gguf",
    "path/to/projection.gguf"
)

# Generate text from image
result = generate_mtmd_with_sampling(
    "Describe this image: <__media__>",
    ["path/to/image.jpg"],
    model=model,
    model_context=model_context,
    vocab=vocab,
    mtmd_context=mtmd_context
)
```

## Enhanced Sampling

```julia
# Use different sampling strategies
creative_result = enhanced_constrained_generation(
    prompt,
    schema,
    tokenizer,
    model_context=model_context,
    vocab=vocab,
    sampling_params=creative_params()
)

# Or create custom sampling parameters
custom_params = SamplingParams(
    temperature=0.9f0,
    top_k=50,
    top_p=0.95f0,
    repeat_penalty=1.1f0
)
```

## LoRA Training and Adaptation

```julia
# Train a LoRA adapter from scratch
gguf_path = train_lora_to_gguf(
    "google/gemma-2-2b-it",
    "your_hf_token_here",
    output_dir="my_lora_models"
)

# Load and use LoRA adapter
model, model_context, vocab = load_and_initialize("path/to/model.gguf")
adapter = load_lora_adapter(model, gguf_path)

# Apply adapter to context
result = set_adapter_lora(model_context, adapter, 1.0f0)
if result == 0
    println("LoRA adapter applied successfully")
end

# Generate with LoRA adaptation
response = generate_with_sampling(
    "Solve this economics problem: ...",
    model=model,
    model_context=model_context,
    vocab=vocab,
    max_new_tokens=100
)

# Remove adapter when done
rm_adapter_lora(model_context, adapter)
free_lora_adapter(adapter)
```

## LoRA Manager for Multiple Adapters

```julia
# Create LoRA manager for handling multiple adapters
manager = LoRAManager(model)

# Load multiple adapters
load_adapter!(manager, "economics", "path/to/econ_adapter.gguf")
load_adapter!(manager, "creative", "path/to/creative_adapter.gguf")

# List available adapters
adapters = list_adapters(manager)
println("Available adapters: ", adapters)

# Switch between adapters
switch_adapter!(manager, model_context, "economics", scale=1.0f0)
# ... generate economics content

switch_adapter!(manager, model_context, "creative", scale=0.8f0)
# ... generate creative content

# Clear active adapter
clear_active_adapter!(manager, model_context)

# Clean up all adapters
free_all_adapters!(manager)
```

## API Reference

### Core Functions

- `init_apis!()`: Initialize all API libraries (Windows)
- `init_apis_not_windows!()`: Initialize all API libraries (Linux/macOS)
- `load_and_initialize(model_path; ...)`: Load LLaMA model for text generation
- `load_and_initialize_mtmd(model_path, proj_path; ...)`: Load multimodal model
- `generate_with_sampling(prompt; ...)`: Basic text generation
- `generate_mtmd_with_sampling(prompt, img_paths; ...)`: Multimodal generation

### Constrained Generation

- `greedy_constrained_generation(prompt, schema, tokenizer; ...)`: Greedy constrained output
- `enhanced_constrained_generation(prompt, schema, tokenizer; ...)`: Enhanced sampling with constraints
- `greedy_mtmd_constrained_generation(prompt, img_paths, schema, tokenizer; ...)`: Multimodal constrained output
- `enhanced_mtmd_constrained_generation(prompt, img_paths, schema, tokenizer; ...)`: Enhanced multimodal sampling

### Sampling Parameters

- `SamplingParams(; ...)`: Create custom sampling configuration
- `creative_params()`: High creativity settings
- `balanced_params()`: Balanced settings (default)
- `focused_params()`: Low creativity, focused output
- `greedy_params()`: Deterministic output

### LoRA Functions

- `train_lora_to_gguf(model_name, hf_token; ...)`: Train LoRA adapter and convert to GGUF
- `load_lora_adapter(model, adapter_path)`: Load LoRA adapter from GGUF file
- `free_lora_adapter(adapter)`: Free LoRA adapter from memory
- `set_adapter_lora(ctx, adapter, scale)`: Set LoRA adapter on context
- `rm_adapter_lora(ctx, adapter)`: Remove LoRA adapter from context
- `clear_adapter_lora(ctx)`: Clear all LoRA adapters from context

### LoRA Manager

- `LoRAManager(model)`: Create LoRA manager for multiple adapters
- `load_adapter!(manager, name, path)`: Load adapter into manager
- `switch_adapter!(manager, ctx, name; scale=1.0f0)`: Switch to specific adapter
- `clear_active_adapter!(manager, ctx)`: Clear active adapter
- `free_all_adapters!(manager)`: Free all managed adapters
- `list_adapters(manager)`: List all loaded adapters
- `get_active_adapter(manager)`: Get name of active adapter
- `list_lora_gguf_files(directory="lora_gguf")`: List GGUF files in directory

## Requirements

### System Dependencies

- **LLaMA.cpp**: Binary builds in `vendors/llama.cpp/build/bin/Release/`
  - `llama.dll` (Windows) or `libllama.so` (Linux)
  - `mtmd.dll` (Windows) or `libmtmd.so` (Linux)
- **Outlines-core**: Binary builds in `vendors/outlines-core/target/release/`
  - `outlines_core.dll` (Windows) or `liboutlines_core.so` (Linux)

⚠️ **Linux/macOS Users**: This package currently includes Windows binaries only. Linux and macOS users need to build the native libraries themselves.

### Authentication Requirements

- **HuggingFace Token**: Required for full API functionality
  - Needed for constrained generation with HuggingFace tokenizers
  - Required for LoRA training with gated models (e.g., Gemma, Llama)
  - Required for vocabulary creation from HuggingFace models
  - Get your token at: https://huggingface.co/settings/tokens
  - Use with `hf_token` parameter in relevant functions

### Optional Dependencies

For image processing (multimodal functionality):
```julia
] add Images FileIO ImageMagick
```

For audio processing:
```julia
] add LibSndFile
```

For LoRA training (automatically installed by `train_lora_to_gguf`):
```bash
pip install torch transformers peft datasets huggingface_hub python-dotenv accelerate
```

## Linux/macOS Build Instructions

Since this package currently includes Windows binaries only, Linux and macOS users need to build the native libraries. The vendor source code and build systems are included.

### Building the Required Binaries

**You must build these binaries BEFORE using the API.** The package will not work without them.

#### Step 1: Build LLaMA.cpp Libraries

**Prerequisites:**
- CMake (3.14+)
- C++ compiler (GCC/Clang)

**Build Commands:**
```bash
cd vendors/llama.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SHARED=ON -DLLAMA_BUILD_SERVER=OFF
make -j$(nproc)
```

**Expected Output:**
- `build/bin/Release/libllama.so` (Linux) or `build/bin/Release/libllama.dylib` (macOS)
- `build/bin/Release/libmtmd.so` (Linux) or `build/bin/Release/libmtmd.dylib` (macOS)

#### Step 2: Build Outlines-core Library

**Prerequisites:**
- Rust/Cargo (install from https://rustup.rs/)

**Build Commands:**
```bash
cd vendors/outlines-core
cargo build --release
```

**Expected Output:**
- `target/release/liboutlines_core.so` (Linux) or `target/release/liboutlines_core.dylib` (macOS)

#### Step 3: Verify Build Success

Check that all required libraries exist:
```bash
# Check LLaMA.cpp libraries
ls -la vendors/llama.cpp/build/bin/Release/libllama.*
ls -la vendors/llama.cpp/build/bin/Release/libmtmd.*

# Check Outlines-core library  
ls -la vendors/outlines-core/target/release/liboutlines_core.*
```

You should see the appropriate `.so` (Linux) or `.dylib` (macOS) files.

### Alternative: Using BinaryBuilder.jl

If you prefer to build from within Julia:

```julia
] add BinaryBuilder

# Build LLaMA.cpp
cd("vendors/llama.cpp")
run(`cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SHARED=ON`)
run(`cmake --build build --config Release`)

# Build Outlines-core
cd("vendors/outlines-core")
run(`cargo build --release`)
```

### Using the Package After Building

After building the required binaries, you can use the package with the special Linux/macOS initialization function:

```julia
using LlamaCppOutlines

# For Linux/macOS users (after building binaries)
init_apis_not_windows!()

# Then use the API normally
model, model_context, vocab = load_and_initialize("path/to/model.gguf")
result = generate_with_sampling("Hello world!", model=model, model_context=model_context, vocab=vocab)
```

**The `init_apis_not_windows!()` function:**
- Automatically detects your platform (Linux vs macOS)
- Looks for the correct library files:
  - Linux: `libllama.so`, `libmtmd.so`, `liboutlines_core.so`
  - macOS: `libllama.dylib`, `libmtmd.dylib`, `liboutlines_core.dylib`
- Provides helpful error messages if libraries are not found
- Shows loaded library paths for verification

**Note:** Windows users should continue using `init_apis!()` instead.

Once initialized, all functionality works identically to Windows.

## Directory Structure

```
LlamaCppOutlines/
├── src/
│   ├── LlamaCppOutlines.jl      # Main module
│   └── constrained_generation.jl # Constrained generation functions
├── lib/
│   ├── llama_api.jl            # LLaMA API bindings
│   ├── mtmd_api.jl             # Multimodal API bindings
│   ├── outlines_api.jl         # Outlines API bindings
│   └── lora_to_gguf.py         # LoRA training script
├── test/
│   ├── runtests.jl             # Test runner
│   ├── test_basic_api.jl       # Basic API tests
│   ├── test_multimodal_api.jl  # Multimodal tests
│   ├── test_outlines_api.jl    # Outlines tests
│   └── test_integration.jl     # Integration tests
├── vendors/
│   ├── llama.cpp/              # LLaMA.cpp source and binaries
│   └── outlines-core/          # Outlines-core source and binaries
└── Project.toml
```

## Testing

```julia
] test LlamaCppOutlines
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License.