module LlamaCppOutlines

# Import the binding modules from lib
include("../lib/llama_api.jl")
include("../lib/mtmd_api.jl")
include("../lib/outlines_api.jl")

import .LlamaCppAPI as llama
import .MtmdCppAPI as mtmd
import .OutlinesCppAPI as outlines

using JSON3, Random, CUDA, Pkg.Artifacts

# Export main user-facing functions
export load_and_initialize, load_and_initialize_mtmd, init_sampler_chain
export generate_with_sampling, generate_mtmd_with_sampling
export greedy_constrained_generation, enhanced_constrained_generation
export greedy_mtmd_constrained_generation, enhanced_mtmd_constrained_generation
export SamplingParams, creative_params, balanced_params, focused_params, greedy_params
export init_apis!

# Export LoRA functions
export load_lora_adapter, free_lora_adapter, set_adapter_lora, rm_adapter_lora, clear_adapter_lora
export LoRAManager, load_adapter!, switch_adapter!, clear_active_adapter!, free_all_adapters!
export list_adapters, get_active_adapter

# Export LoRA training functions
export train_lora_to_gguf, list_lora_gguf_files

# Initialize APIs with proper vendor paths
"""
    init_apis!()

Initialize LlamaCppOutlines APIs with automatic binary management and GPU detection.
This function automatically downloads and configures the appropriate binaries for your system:

- **Windows x64 (CPU)**: Downloads CPU-optimized llama.cpp binaries
- **Windows x64 (GPU)**: Downloads CUDA-accelerated binaries when NVIDIA GPU is detected
- **Auto-detection**: Automatically selects GPU binaries if CUDA is functional, falls back to CPU otherwise

# Features

- **Zero configuration**: No manual binary building required
- **Automatic GPU detection**: Uses `CUDA.functional()` to detect GPU support
- **Artifact management**: Downloads and caches binaries automatically via Julia's artifact system
- **Complete toolset**: Includes all llama.cpp utilities (CLI, server, quantization, fine-tuning, LoRA export, etc.)
- **Outlines integration**: Includes outlines-core for structured generation

# Example

```julia
using LlamaCppOutlines

# Initialize APIs (downloads binaries automatically)
init_apis!()

# APIs are now ready to use
model, model_context, vocab = load_and_initialize("model.gguf")
```

# Platform Support

- ✅ **Windows x64**: Fully supported with automatic binary downloads
- 🔄 **Linux/macOS**: Coming in future releases

# GPU Requirements

For GPU acceleration, ensure you have:
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- CUDA.jl working: `using CUDA; CUDA.functional()` should return `true`

# Notes

- Binaries are cached locally after first download
- Will display whether GPU or CPU binaries are being used
- No internet connection required after initial download
- Falls back gracefully to CPU binaries if GPU detection fails
"""
function init_apis!()
    println("🔧 Initializing LlamaCppOutlines APIs...")
    
    # Get artifact paths
    artifacts_toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    
    # Force download artifacts upfront
    println("📦 Ensuring binaries are available...")
    
    # Always ensure CPU binaries are available (fallback)
    try
        print("   Downloading CPU binaries... ")
        ensure_artifact_installed("LlamaCppOutlines_CPU", artifacts_toml)
        println("✅")
    catch e
        println("❌")
        @warn "Failed to download CPU binaries" exception=e
        error("Cannot proceed without CPU binaries. Please check your internet connection.")
    end
    
    # Determine which artifact to use
    artifact_name = "LlamaCppOutlines_CPU"  # Default to CPU
    
    # Try to detect and download GPU support
    gpu_available = false
    try
        if CUDA.functional()
            print("   GPU detected, downloading CUDA binaries... ")
            ensure_artifact_installed("LlamaCppOutlines_GPU", artifacts_toml)
            println("✅")
            
            gpu_hash = artifact_hash("LlamaCppOutlines_GPU", artifacts_toml)
            if gpu_hash !== nothing && artifact_exists(gpu_hash)
                artifact_name = "LlamaCppOutlines_GPU"
                gpu_available = true
                println("🚀 Using GPU-accelerated binaries")
            end
        else
            println("ℹ️  No GPU detected, using CPU binaries")
        end
    catch e
        println("⚠️  GPU detection failed, falling back to CPU binaries")
        @warn "GPU setup failed" exception=e
    end
    
    # Get the selected artifact path
    hash = artifact_hash(artifact_name, artifacts_toml)
    if hash === nothing
        error("❌ Selected binaries not found in artifacts")
    end
    
    artifact_dir = artifact_path(hash)
    bin_path = joinpath(artifact_dir, "bin")
    
    # Verify binaries exist
    llama_dll = joinpath(bin_path, "llama.dll")
    mtmd_dll = joinpath(bin_path, "mtmd.dll")
    outlines_dll = joinpath(bin_path, "outlines_core.dll")
    
    for (name, path) in [("llama.dll", llama_dll), ("mtmd.dll", mtmd_dll), ("outlines_core.dll", outlines_dll)]
        if !isfile(path)
            error("❌ Required binary not found: $name at $path")
        end
    end
    
    # Initialize APIs
    try
        print("🔗 Loading LLaMA API... ")
        llama.init!(llama_dll)
        println("✅")
        
        print("🔗 Loading MTMD API... ")
        mtmd.init!(mtmd_dll)
        println("✅")
        
        print("🔗 Loading Outlines API... ")
        outlines.init!(outlines_dll)
        println("✅")
        
        if gpu_available
            println("🎉 All APIs initialized successfully with GPU acceleration!")
        else
            println("🎉 All APIs initialized successfully with CPU!")
        end
        
    catch e
        error("❌ Failed to load APIs: $e")
    end
end

"""
    init_apis_not_windows!()

Initialize LlamaCppOutlines APIs for Linux/macOS platforms using platform-specific library files.

This function automatically detects your platform and loads the appropriate shared libraries:
- Linux: .so files (libllama.so, libmtmd.so, liboutlines_core.so)
- macOS: .dylib files (libllama.dylib, libmtmd.dylib, liboutlines_core.dylib)

# Requirements
Before calling this function, you must build the required binaries:
1. Build LLaMA.cpp libraries in vendors/llama.cpp/build/bin/Release/
2. Build Outlines-core library in vendors/outlines-core/target/release/

# Example
```julia
# For Linux/macOS users after building binaries
init_apis_not_windows!()

# Then use the API normally
model, model_context, vocab = load_and_initialize("model.gguf")
```

# Notes
- Windows users should use init_apis!() instead
- Will show clear error messages if libraries are not found
- Displays loaded library paths for verification
"""
function init_apis_not_windows!()
    @warn "init_apis_not_windows! is deprecated. Non-Windows support will be added in future releases, the current support is only Windows."

    # Determine library extension and prefix based on platform
    if Sys.islinux()
        lib_ext = ".so"
        lib_prefix = "lib"
        platform_name = "Linux"
    elseif Sys.isapple()
        lib_ext = ".dylib"
        lib_prefix = "lib"
        platform_name = "macOS"
    else
        error("Unsupported platform: $(Sys.MACHINE)\n" *
              "This function is for Linux/macOS only. Windows users should use init_apis!().")
    end

    println("   Platform detected: $platform_name")

    # Initialize LLaMA API
    llama_lib = joinpath(@__DIR__, "..", "vendors", "llama.cpp", "build", "bin", "Release", "$(lib_prefix)llama$(lib_ext)")
    if !isfile(llama_lib)
        error("LLaMA library not found: $llama_lib\n" *
              "Please build llama.cpp first:\n" *
              "  cd vendors/llama.cpp\n" *
              "  mkdir -p build && cd build\n" *
              "  cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SHARED=ON -DLLAMA_BUILD_SERVER=OFF\n" *
              "  make -j\$(nproc)")
    end
    llama.init!(llama_lib)

    # Initialize MTMD API  
    mtmd_lib = joinpath(@__DIR__, "..", "vendors", "llama.cpp", "build", "bin", "Release", "$(lib_prefix)mtmd$(lib_ext)")
    if !isfile(mtmd_lib)
        error("MTMD library not found: $mtmd_lib\n" *
              "Please build llama.cpp with multimodal support first:\n" *
              "  cd vendors/llama.cpp\n" *
              "  mkdir -p build && cd build\n" *
              "  cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SHARED=ON -DLLAMA_BUILD_SERVER=OFF\n" *
              "  make -j\$(nproc)")
    end
    mtmd.init!(mtmd_lib)

    # Initialize Outlines API
    outlines_lib = joinpath(@__DIR__, "..", "vendors", "outlines-core", "target", "release", "$(lib_prefix)outlines_core$(lib_ext)")
    if !isfile(outlines_lib)
        error("Outlines library not found: $outlines_lib\n" *
              "Please build outlines-core first:\n" *
              "  cd vendors/outlines-core\n" *
              "  cargo build --release")
    end
    outlines.init!(outlines_lib)

    println("✅ All APIs initialized successfully for $platform_name!")
    println("   📚 LLaMA: $llama_lib")
    println("   🖼️  MTMD: $mtmd_lib")
    println("   🎯 Outlines: $outlines_lib")
end

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading and Initialization Functions
# ─────────────────────────────────────────────────────────────────────────────

const LLAMA_DEFAULT_SEED = 0xDEADBEEF

"""
    load_and_initialize(model_path::String; seed::Integer=LLAMA_DEFAULT_SEED, n_ctx::Integer=16184, n_batch::Integer=4096, n_threads::Integer=6)

Load and initialize a LLaMA model for text generation.

# Arguments
- `model_path::String`: Path to the GGUF model file

# Keyword Arguments
- `seed::Integer=LLAMA_DEFAULT_SEED`: Random seed for reproducibility
- `n_ctx::Integer=16184`: Context size
- `n_batch::Integer=4096`: Batch size
- `n_threads::Integer=6`: Number of threads

# Returns
- `Tuple{Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}}`: (model, model_context, vocab)

# Example
```julia
model, model_context, vocab = load_and_initialize("path/to/model.gguf")
```
"""
function load_and_initialize(model_path::String;
    seed::Integer=LLAMA_DEFAULT_SEED,
    n_ctx::Integer=16184,
    n_batch::Integer=4096,
    n_threads::Integer=6)

    default_params = llama.model_default_params()
    params = llama.llama_model_params(
        default_params.devices,
        default_params.tensor_buft_overrides,
        Cint(999),              # offload to GPU as many layers as possible
        default_params.split_mode,
        default_params.main_gpu,
        default_params.tensor_split,
        default_params.progress_callback,
        default_params.progress_callback_user_data,
        default_params.kv_overrides,
        default_params.vocab_only,
        UInt8(1),               # use_mmap
        default_params.use_mlock,
        default_params.check_tensors,
        (0x00, 0x00, 0x00, 0x00)
    )

    model = llama.load_model_from_file(model_path, params)
    @assert model != C_NULL

    vocab = llama.get_vocab(model)
    @assert vocab != C_NULL

    context = llama.context_default_params()
    context.n_ctx = n_ctx
    context.n_batch = n_batch
    context.embeddings = false
    context.n_threads = n_threads

    model_context = llama.new_context_with_model(model, context)
    @assert model_context != C_NULL

    return (model, model_context, vocab)
end

"""
    load_and_initialize_mtmd(model_path::String, proj_path::String; seed::Integer=LLAMA_DEFAULT_SEED, n_ctx::Integer=16184, n_batch::Integer=4096, n_threads::Integer=6, use_gpu::Bool=false, print_timings::Bool=false, verbosity::Integer=1)

Load and initialize a multimodal LLaMA model for text and image generation.

# Arguments
- `model_path::String`: Path to the GGUF model file
- `proj_path::String`: Path to the multimodal projection file

# Keyword Arguments
- `seed::Integer=LLAMA_DEFAULT_SEED`: Random seed for reproducibility
- `n_ctx::Integer=16184`: Context size
- `n_batch::Integer=4096`: Batch size
- `n_threads::Integer=6`: Number of threads
- `use_gpu::Bool=false`: Whether to use GPU
- `print_timings::Bool=false`: Whether to print timing information
- `verbosity::Integer=1`: Verbosity level

# Returns
- `Tuple{Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}}`: (model, model_context, mtmd_context, vocab)

# Example
```julia
model, model_context, mtmd_context, vocab = load_and_initialize_mtmd("model.gguf", "projection.gguf")
```
"""
function load_and_initialize_mtmd(model_path::String, proj_path::String;
    seed::Integer=LLAMA_DEFAULT_SEED,
    n_ctx::Integer=16184,
    n_batch::Integer=4096,
    n_threads::Integer=6,
    use_gpu::Bool=false,
    print_timings::Bool=false,
    verbosity::Integer=1)

    # Load model and create context
    model, model_context, vocab = load_and_initialize(model_path;
        seed=seed,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_threads=n_threads)
    @assert model != C_NULL "Failed to load model"
    @assert model_context != C_NULL "Failed to create model context"
    @assert vocab != C_NULL "Failed to get vocabulary"

    # Initialize multimodal context
    mtmd_params = mtmd.context_params_default()
    mtmd_params.use_gpu = use_gpu
    mtmd_params.print_timings = print_timings
    mtmd_params.n_threads = n_threads
    mtmd_params.verbosity = verbosity

    mtmd_context = mtmd.init_from_file(proj_path, model, mtmd_params)
    @assert mtmd_context != C_NULL "Failed to initialize multimodal context"

    return (model, model_context, mtmd_context, vocab)
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

function init_sampler_chain(; k=50, p=0.9, t=0.7, penalty_last_n=64, penalty_repeat=1.2, penalty_freq=0.5, penalty_present=0.5,
    seed=LLAMA_DEFAULT_SEED)
    dist = llama.sampler_init_dist(seed)

    sparams = llama.sampler_chain_default_params()
    chain = llama.sampler_chain_init(sparams)
    topk = llama.sampler_init_top_k(k)
    topp = llama.sampler_init_top_p(p, UInt(1))
    temp = llama.sampler_init_temp(t)
    pen = llama.sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present)

    llama.sampler_chain_add(chain, topk)
    llama.sampler_chain_add(chain, topp)
    llama.sampler_chain_add(chain, temp)
    llama.sampler_chain_add(chain, pen)
    llama.sampler_chain_add(chain, dist)

    return chain
end

function sample_token(chain::Ptr{Cvoid}, ctx::Ptr{Cvoid}, idx::Integer)::Integer
    # Sample a token from the chain
    tok = llama.sampler_sample(chain, ctx, idx)
    # Accept the sampled token
    llama.sampler_accept(chain, tok)
    return tok
end

function load_an_image_bitmap(path::String)
    # This function would need Images.jl, FileIO.jl, ImageMagick.jl
    # For now, we'll include a placeholder that users can implement
    error("load_an_image_bitmap requires Images.jl, FileIO.jl, and ImageMagick.jl to be loaded. Please see documentation for implementation.")
end

function clear_context_state(llama_ctx)
    # Clear KV cache and reset context state
    llama.kv_cache_clear(llama_ctx)
    llama.set_embeddings(llama_ctx, false)
    println("🧹 Context state cleared")
end

# ─────────────────────────────────────────────────────────────────────────────
# Basic Generation Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_with_sampling(prompt::String; model, model_context, vocab, max_new_tokens::Integer=50, k=50, p=0.9, t=0.7, penalty_last_n=64, penalty_repeat=1.2, penalty_freq=0.5, penalty_present=0.5, seed=LLAMA_DEFAULT_SEED)

Generate text using sampling with the LLaMA model.

# Arguments
- `prompt::String`: Input prompt for generation

# Keyword Arguments
- `model`: Pre-loaded model
- `model_context`: Pre-loaded model context
- `vocab`: Pre-loaded vocabulary
- `max_new_tokens::Integer=50`: Maximum number of tokens to generate
- `k=50`: Top-k sampling parameter
- `p=0.9`: Top-p sampling parameter
- `t=0.7`: Temperature parameter
- `penalty_last_n=64`: Penalty lookback window
- `penalty_repeat=1.2`: Repetition penalty
- `penalty_freq=0.5`: Frequency penalty
- `penalty_present=0.5`: Presence penalty
- `seed=LLAMA_DEFAULT_SEED`: Random seed

# Returns
- `String`: Generated text

# Example
```julia
result = generate_with_sampling("What is the capital of France?", 
    model=model, model_context=model_context, vocab=vocab)
```
"""
function generate_with_sampling(prompt::String;
    model,
    model_context,
    vocab,
    max_new_tokens::Integer=50,
    k=50, p=0.9, t=0.7,
    penalty_last_n=64, penalty_repeat=1.2,
    penalty_freq=0.5, penalty_present=0.5,
    seed=LLAMA_DEFAULT_SEED)

    # Clear KV cache and disable embeddings
    llama.kv_cache_clear(model_context)
    llama.set_embeddings(model_context, false)

    chain = init_sampler_chain(k=k, p=p, t=t,
        penalty_last_n=penalty_last_n, penalty_repeat=penalty_repeat,
        penalty_freq=penalty_freq, penalty_present=penalty_present,
        seed=seed)

    # Tokenize and decode a prompt
    prompt_tokens = llama.tokenize(vocab, prompt)
    batch, keep = llama.build_batch(prompt_tokens)
    try
        @assert llama.decode(model_context, batch)
    catch e
        println("Error during decoding: ", e)
    end

    # Collect generated tokens
    vocab_size = llama.vocab_n_tokens(vocab)
    eos_id = llama.token_eos(vocab)

    all_tokens = copy(prompt_tokens)
    current_pos = length(all_tokens)
    generated = ""

    for step in 1:max_new_tokens
        next_tok = sample_token(chain, model_context, Cint(-1))
        if next_tok == eos_id
            break
        end
        generated *= llama.vocab_get_text(vocab, next_tok)
        batch, keep = llama.build_single_token_batch(next_tok, Cint(current_pos))
        @assert llama.decode(model_context, batch)
        push!(all_tokens, next_tok)
        current_pos += 1
    end

    # Clean up
    llama.sampler_free(chain)
    return generated
end

"""
    generate_mtmd_with_sampling(prompt::String, img_paths::Vector{String}; model, model_context, vocab, mtmd_context, max_new_tokens::Integer=50, k=50, p=0.9, t=0.7, penalty_last_n=64, penalty_repeat=1.2, penalty_freq=0.5, penalty_present=0.5, seed=LLAMA_DEFAULT_SEED)

Generate multimodal text using sampling with images.

# Arguments
- `prompt::String`: Input prompt for generation
- `img_paths::Vector{String}`: Paths to image files

# Keyword Arguments
- `model`: Pre-loaded model
- `model_context`: Pre-loaded model context
- `vocab`: Pre-loaded vocabulary
- `mtmd_context`: Pre-loaded multimodal context
- `max_new_tokens::Integer=50`: Maximum number of tokens to generate
- `k=50`: Top-k sampling parameter
- `p=0.9`: Top-p sampling parameter
- `t=0.7`: Temperature parameter
- `penalty_last_n=64`: Penalty lookback window
- `penalty_repeat=1.2`: Repetition penalty
- `penalty_freq=0.5`: Frequency penalty
- `penalty_present=0.5`: Presence penalty
- `seed=LLAMA_DEFAULT_SEED`: Random seed

# Returns
- `String`: Generated text

# Example
```julia
result = generate_mtmd_with_sampling("Describe this image: <__media__>", ["image.jpg"],
    model=model, model_context=model_context, vocab=vocab, mtmd_context=mtmd_context)
```
"""
function generate_mtmd_with_sampling(prompt::String, img_paths::Vector{String};
    model,
    model_context,
    vocab,
    mtmd_context,
    max_new_tokens::Integer=50,
    k=50, p=0.9, t=0.7,
    penalty_last_n=64, penalty_repeat=1.2,
    penalty_freq=0.5, penalty_present=0.5,
    seed=LLAMA_DEFAULT_SEED)

    # Clear KV cache and disable embeddings
    clear_context_state(model_context)

    image_bitmaps = load_an_image_bitmap.(img_paths)

    chunks = mtmd.input_chunks_init()

    prompt_text = mtmd.create_input_text(prompt)

    ret = mtmd.tokenize(mtmd_context, chunks, prompt_text, image_bitmaps)
    if ret != 0
        error("Multimodal tokenization failed with code $ret")
    end

    chain = init_sampler_chain(k=k, p=p, t=t,
        penalty_last_n=penalty_last_n, penalty_repeat=penalty_repeat,
        penalty_freq=penalty_freq, penalty_present=penalty_present,
        seed=seed)

    n_past = 0
    seq_id = 0
    n_batch = llama.n_batch(model_context)

    success, new_n_past = mtmd.helper_eval_chunks(mtmd_context, model_context, chunks,
        n_past, seq_id, n_batch, true)

    eos_id = llama.token_eos(vocab)
    current_pos = new_n_past
    generated = ""

    for step in 1:max_new_tokens
        next_tok = sample_token(chain, model_context, Cint(-1))
        if next_tok == eos_id
            break
        end
        generated *= llama.vocab_get_text(vocab, next_tok)
        batch, keep = llama.build_single_token_batch(next_tok, Cint(current_pos))
        @assert llama.decode(model_context, batch)
        current_pos += 1
    end

    # Clean up
    llama.sampler_free(chain)
    mtmd.input_chunks_free(chunks)
    for bitmap in image_bitmaps
        mtmd.bitmap_free(bitmap)
    end
    return generated
end

# ─────────────────────────────────────────────────────────────────────────────
# Sampling Parameters Structure
# ─────────────────────────────────────────────────────────────────────────────

"""
Sampling parameters for controlled variety in constrained generation.
"""
mutable struct SamplingParams
    # Temperature sampling
    temperature::AbstractFloat         # 0.0 = greedy, 1.0 = neutral, >1.0 = more random

    # Top-k sampling  
    top_k::Integer                # Only sample from top-k tokens (0 = disabled)

    # Top-p (nucleus) sampling
    top_p::AbstractFloat              # Cumulative probability threshold (1.0 = disabled)

    # Min-p sampling
    min_p::AbstractFloat              # Minimum probability relative to max (0.0 = disabled)

    # Repetition penalty
    repeat_penalty::AbstractFloat     # Penalty for repeated tokens (1.0 = disabled)
    repeat_last_n::Integer        # Look back N tokens for repetition (0 = disabled)

    # Random seed control
    seed::Integer                 # Random seed (-1 = use current time)

    # Debugging
    verbose_sampling::Bool      # Print sampling details
end

# Default conservative parameters
function SamplingParams(;
    temperature::AbstractFloat=0.8f0,
    top_k::Integer=40,
    top_p::AbstractFloat=0.9f0,
    min_p::AbstractFloat=0.05f0,
    repeat_penalty::AbstractFloat=1.1f0,
    repeat_last_n::Integer=64,
    seed::Integer=-1,
    verbose_sampling::Bool=false
)
    return SamplingParams(temperature, top_k, top_p, min_p, repeat_penalty, repeat_last_n, seed, verbose_sampling)
end

# Preset configurations
creative_params() = SamplingParams(temperature=1.2f0, top_k=100, top_p=0.95f0, min_p=0.02f0)
balanced_params() = SamplingParams(temperature=0.8f0, top_k=40, top_p=0.9f0, min_p=0.05f0)
focused_params() = SamplingParams(temperature=0.3f0, top_k=20, top_p=0.8f0, min_p=0.1f0)
greedy_params() = SamplingParams(temperature=0.0f0, top_k=1, top_p=1.0f0, min_p=0.0f0)

# ─────────────────────────────────────────────────────────────────────────────
# LoRA Functions (forwarded from llama module)
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_lora_adapter(model, adapter_path::String)

Load a LoRA adapter from a GGUF file.

# Arguments
- `model`: Pre-loaded LLaMA model
- `adapter_path::String`: Path to the LoRA adapter GGUF file

# Returns
- Adapter handle for use with other LoRA functions

# Example
```julia
adapter = load_lora_adapter(model, "path/to/adapter.gguf")
```
"""
load_lora_adapter(model, adapter_path::String) = llama.load_lora_adapter(model, adapter_path)

"""
    free_lora_adapter(adapter)

Free a LoRA adapter from memory.

# Arguments
- `adapter`: Adapter handle returned by load_lora_adapter

# Example
```julia
free_lora_adapter(adapter)
```
"""
free_lora_adapter(adapter) = llama.free_lora_adapter(adapter)

"""
    set_adapter_lora(ctx, adapter, scale::Float32)

Set a LoRA adapter on a context with specified scale.

# Arguments
- `ctx`: Model context
- `adapter`: Adapter handle
- `scale::Float32`: Scale factor (0.0 to 1.0, typically 1.0)

# Returns
- Integer result code (0 for success)

# Example
```julia
result = set_adapter_lora(model_context, adapter, 1.0f0)
```
"""
set_adapter_lora(ctx, adapter, scale::Float32) = llama.set_adapter_lora(ctx, adapter, scale)

"""
    rm_adapter_lora(ctx, adapter)

Remove a LoRA adapter from a context.

# Arguments
- `ctx`: Model context
- `adapter`: Adapter handle

# Returns
- Integer result code (0 for success)

# Example
```julia
result = rm_adapter_lora(model_context, adapter)
```
"""
rm_adapter_lora(ctx, adapter) = llama.rm_adapter_lora(ctx, adapter)

"""
    clear_adapter_lora(ctx)

Clear all LoRA adapters from a context.

# Arguments
- `ctx`: Model context

# Returns
- Integer result code (0 for success)

# Example
```julia
result = clear_adapter_lora(model_context)
```
"""
clear_adapter_lora(ctx) = llama.clear_adapter_lora(ctx)

"""
    LoRAManager(model)

Create a LoRA manager for handling multiple adapters.

# Arguments
- `model`: Pre-loaded LLaMA model

# Returns
- LoRAManager instance

# Example
```julia
manager = LoRAManager(model)
```
"""
LoRAManager(model) = llama.LoRAManager(model)

"""
    load_adapter!(manager::LoRAManager, name::String, path::String)

Load an adapter into the LoRA manager.

# Arguments
- `manager::LoRAManager`: LoRA manager instance
- `name::String`: Name to assign to the adapter
- `path::String`: Path to the adapter GGUF file

# Example
```julia
load_adapter!(manager, "economics", "path/to/econ_adapter.gguf")
```
"""
load_adapter!(manager, name::String, path::String) = llama.load_adapter!(manager, name, path)

"""
    switch_adapter!(manager::LoRAManager, ctx, name::String; scale::Float32=1.0f0)

Switch to a specific adapter in the LoRA manager.

# Arguments
- `manager::LoRAManager`: LoRA manager instance
- `ctx`: Model context
- `name::String`: Name of the adapter to activate
- `scale::Float32=1.0f0`: Scale factor for the adapter

# Example
```julia
switch_adapter!(manager, model_context, "economics", scale=1.0f0)
```
"""
switch_adapter!(manager, ctx, name::String; scale::Float32=1.0f0) = llama.switch_adapter!(manager, ctx, name, scale=scale)

"""
    clear_active_adapter!(manager::LoRAManager, ctx)

Clear the currently active adapter from context.

# Arguments
- `manager::LoRAManager`: LoRA manager instance
- `ctx`: Model context

# Example
```julia
clear_active_adapter!(manager, model_context)
```
"""
clear_active_adapter!(manager, ctx) = llama.clear_active_adapter!(manager, ctx)

"""
    free_all_adapters!(manager::LoRAManager)

Free all adapters managed by the LoRA manager.

# Arguments
- `manager::LoRAManager`: LoRA manager instance

# Example
```julia
free_all_adapters!(manager)
```
"""
free_all_adapters!(manager) = llama.free_all_adapters!(manager)

"""
    list_adapters(manager::LoRAManager)

List all adapters loaded in the LoRA manager.

# Arguments
- `manager::LoRAManager`: LoRA manager instance

# Returns
- List of adapter names

# Example
```julia
adapters = list_adapters(manager)
println("Available adapters: ", adapters)
```
"""
list_adapters(manager) = llama.list_adapters(manager)

"""
    get_active_adapter(manager::LoRAManager)

Get the name of the currently active adapter.

# Arguments
- `manager::LoRAManager`: LoRA manager instance

# Returns
- String name of active adapter, or empty string if none

# Example
```julia
active = get_active_adapter(manager)
if !isempty(active)
    println("Active adapter: ", active)
end
```
"""
get_active_adapter(manager) = llama.get_active_adapter(manager)

# Include the constrained generation functions
include("constrained_generation.jl")

# Include the LoRA training functions  
include("lora_training.jl")

end # module
