module LlamaCppAPI

using Libdl

# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

@enum LlamaVocabType begin
    LLAMA_VOCAB_TYPE_NONE = 0  # For models without vocab
    LLAMA_VOCAB_TYPE_SPM = 1   # LLaMA byte-level BPE w/ byte fallback
    LLAMA_VOCAB_TYPE_BPE = 2   # GPT-2 byte-level BPE
    LLAMA_VOCAB_TYPE_WPM = 3   # BERT WordPiece
    LLAMA_VOCAB_TYPE_UGM = 4   # T5 Unigram
    LLAMA_VOCAB_TYPE_RWKV = 5  # RWKV greedy tokenizer
end

# ─────────────────────────────────────────────────────────────────────────────
# Structs
# ─────────────────────────────────────────────────────────────────────────────

mutable struct llama_model_params
    devices::Ptr{Cvoid}
    tensor_buft_overrides::Ptr{Cvoid}
    n_gpu_layers::Cint
    split_mode::Cint
    main_gpu::Cint
    tensor_split::Ptr{Cvoid}
    progress_callback::Ptr{Cvoid}
    progress_callback_user_data::Ptr{Cvoid}
    kv_overrides::Ptr{Cvoid}
    vocab_only::UInt8
    use_mmap::UInt8
    use_mlock::UInt8
    check_tensors::UInt8
    pad::NTuple{4,UInt8}
end

mutable struct llama_context_params
    n_ctx::Cuint
    n_batch::Cuint
    n_ubatch::Cuint
    n_seq_max::Cuint
    n_threads::Cint
    n_threads_batch::Cint
    rope_scaling_type::Cint
    pooling_type::Cint
    attention_type::Cint
    rope_freq_base::Cfloat
    rope_freq_scale::Cfloat
    yarn_ext_factor::Cfloat
    yarn_attn_factor::Cfloat
    yarn_beta_fast::Cfloat
    yarn_beta_slow::Cfloat
    yarn_orig_ctx::Cuint
    defrag_thold::Cfloat
    cb_eval::Ptr{Cvoid}
    cb_eval_user_data::Ptr{Cvoid}
    type_k::Cint
    type_v::Cint
    abort_callback::Ptr{Cvoid}
    abort_callback_data::Ptr{Cvoid}
    embeddings::Bool
    offload_kqv::Bool
    flash_attn::Bool
    no_perf::Bool
    op_offload::Bool
    swa_full::Bool
    _padding::NTuple{2,UInt8}
end

struct llama_sampler_chain_params
    no_perf::UInt8
end

mutable struct llama_batch
    n_tokens::Cint              # int32_t
    token::Ptr{Cint}            # llama_token*  (int32_t*)
    embd::Ptr{Cfloat}           # float*
    pos::Ptr{Cint}              # llama_pos*  (int32_t*)
    n_seq_id::Ptr{Cint}         # int32_t*
    seq_id::Ptr{Ptr{Cint}}      # int32_t**
    logits::Ptr{Int8}           # int8_t*
end

struct LlamaLogitBias
    token::Cint
    bias::Float32
end

# ─────────────────────────────────────────────────────────────────────────────
# Global Variables (set by init!)
# ─────────────────────────────────────────────────────────────────────────────

# DLL handle and function pointers - initialized by init!()
global handle::Ptr{Nothing} = C_NULL

# Core model functions
global model_default_params_fn::Ptr{Nothing} = C_NULL
global model_load_fn::Ptr{Nothing} = C_NULL
global model_free_fn::Ptr{Nothing} = C_NULL
global model_get_vocab_fn::Ptr{Nothing} = C_NULL
global model_desc_fn::Ptr{Nothing} = C_NULL

# Context functions
global ctx_default_params_fn::Ptr{Nothing} = C_NULL
global ctx_create_fn::Ptr{Nothing} = C_NULL
global ctx_free_fn::Ptr{Nothing} = C_NULL
global n_batch_fn::Ptr{Nothing} = C_NULL
global n_ctx_fn::Ptr{Nothing} = C_NULL

# Tokenization functions
global tokenize_fn::Ptr{Nothing} = C_NULL
global vocab_type_fn::Ptr{Nothing} = C_NULL
global vocab_n_tokens_fn::Ptr{Nothing} = C_NULL
global vocab_get_text_fn::Ptr{Nothing} = C_NULL
global vocab_eos_fn::Ptr{Nothing} = C_NULL
global vocab_bos_fn::Ptr{Nothing} = C_NULL
global vocab_is_eog_fn::Ptr{Nothing} = C_NULL

# Inference functions
global decode_fn::Ptr{Nothing} = C_NULL
global get_logits_fn::Ptr{Nothing} = C_NULL
global get_logits_ith_fn::Ptr{Nothing} = C_NULL
global set_causal_attn_fn::Ptr{Nothing} = C_NULL

# Sampling functions
global sampler_chain_default_params_fn::Ptr{Nothing} = C_NULL
global sampler_chain_init_fn::Ptr{Nothing} = C_NULL
global sampler_chain_add_fn::Ptr{Nothing} = C_NULL
global sampler_init_topk_fn::Ptr{Nothing} = C_NULL
global sampler_init_topp_fn::Ptr{Nothing} = C_NULL
global sampler_init_temp_fn::Ptr{Nothing} = C_NULL
global sampler_init_penalties_fn::Ptr{Nothing} = C_NULL
global sampler_init_dist_fn::Ptr{Nothing} = C_NULL
global sampler_init_logit_bias_fn::Ptr{Nothing} = C_NULL
global sampler_sample_fn::Ptr{Nothing} = C_NULL
global sampler_accept_fn::Ptr{Nothing} = C_NULL
global sampler_free_fn::Ptr{Nothing} = C_NULL

# KV cache functions
global kv_clear_fn::Ptr{Nothing} = C_NULL
global kv_seq_rm_fn::Ptr{Nothing} = C_NULL
global kv_seq_keep_fn::Ptr{Nothing} = C_NULL
global kv_seq_cp_fn::Ptr{Nothing} = C_NULL
global kv_seq_add_fn::Ptr{Nothing} = C_NULL
global kv_self_n_tokens_fn::Ptr{Nothing} = C_NULL

# Embedding functions
global set_embeddings_fn::Ptr{Nothing} = C_NULL
global get_embeddings_fn::Ptr{Nothing} = C_NULL
global get_embeddings_ith_fn::Ptr{Nothing} = C_NULL

# Model metadata functions
global meta_val_str_fn::Ptr{Nothing} = C_NULL
global meta_count_fn::Ptr{Nothing} = C_NULL
global meta_key_fn::Ptr{Nothing} = C_NULL
global meta_val_fn::Ptr{Nothing} = C_NULL

# LoRA adapter functions
global adapter_lora_init_fn::Ptr{Nothing} = C_NULL
global adapter_lora_free_fn::Ptr{Nothing} = C_NULL
global set_adapter_lora_fn::Ptr{Nothing} = C_NULL
global rm_adapter_lora_fn::Ptr{Nothing} = C_NULL
global clear_adapter_lora_fn::Ptr{Nothing} = C_NULL


# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

"""
    init!(dll_path::String)

Initialize the LlamaCpp module by loading the DLL and resolving function pointers.
Must be called before using any other functions.

# Arguments
- `dll_path::String`: Path to the llama.dll file

# Example
```julia
LlamaCpp.init!("path/to/llama.dll")
```
"""
function init!(dll_path::String)
    global handle = dlopen(dll_path)

    # Core model functions
    global model_default_params_fn = dlsym(handle, :llama_model_default_params)
    global model_load_fn = dlsym(handle, :llama_load_model_from_file)
    global model_free_fn = dlsym(handle, :llama_free_model)
    global model_desc_fn = dlsym(handle, :llama_model_desc)
    global model_get_vocab_fn = dlsym(handle, :llama_model_get_vocab)

    # Context functions
    global ctx_default_params_fn = dlsym(handle, :llama_context_default_params)
    global ctx_create_fn = dlsym(handle, :llama_new_context_with_model)
    global ctx_free_fn = dlsym(handle, :llama_free)
    global n_batch_fn = dlsym(handle, :llama_n_batch)
    global n_ctx_fn = dlsym(handle, :llama_n_ctx)

    # Tokenization functions
    global tokenize_fn = dlsym(handle, :llama_tokenize)
    global vocab_type_fn = dlsym(handle, :llama_vocab_type)
    global vocab_n_tokens_fn = dlsym(handle, :llama_n_vocab)
    global vocab_get_text_fn = dlsym(handle, :llama_vocab_get_text)
    global vocab_eos_fn = dlsym(handle, :llama_token_eos)
    global vocab_bos_fn = dlsym(handle, :llama_token_bos)
    global vocab_is_eog_fn = dlsym(handle, :llama_token_is_eog)

    # Inference functions
    global decode_fn = dlsym(handle, :llama_decode)
    global get_logits_fn = dlsym(handle, :llama_get_logits)
    global get_logits_ith_fn = dlsym(handle, :llama_get_logits_ith)
    global set_causal_attn_fn = dlsym(handle, :llama_set_causal_attn)

    # Sampling functions
    global sampler_chain_default_params_fn = dlsym(handle, :llama_sampler_chain_default_params)
    global sampler_chain_init_fn = dlsym(handle, :llama_sampler_chain_init)
    global sampler_chain_add_fn = dlsym(handle, :llama_sampler_chain_add)
    global sampler_init_topk_fn = dlsym(handle, :llama_sampler_init_top_k)
    global sampler_init_topp_fn = dlsym(handle, :llama_sampler_init_top_p)
    global sampler_init_temp_fn = dlsym(handle, :llama_sampler_init_temp)
    global sampler_init_penalties_fn = dlsym(handle, :llama_sampler_init_penalties)
    global sampler_init_dist_fn = dlsym(handle, :llama_sampler_init_dist)
    global sampler_init_logit_bias_fn = dlsym(handle, :llama_sampler_init_logit_bias)
    global sampler_sample_fn = dlsym(handle, :llama_sampler_sample)
    global sampler_accept_fn = dlsym(handle, :llama_sampler_accept)
    global sampler_free_fn = dlsym(handle, :llama_sampler_free)

    # KV cache functions
    global kv_clear_fn = dlsym(handle, :llama_kv_self_clear)
    global kv_seq_rm_fn = dlsym(handle, :llama_kv_self_seq_rm)
    global kv_seq_keep_fn = dlsym(handle, :llama_kv_self_seq_keep)
    global kv_seq_cp_fn = dlsym(handle, :llama_kv_self_seq_cp)
    global kv_seq_add_fn = dlsym(handle, :llama_kv_self_seq_add)
    global kv_self_n_tokens_fn = dlsym(handle, :llama_kv_self_n_tokens)

    # Embedding functions
    global set_embeddings_fn = dlsym(handle, :llama_set_embeddings)
    global get_embeddings_fn = dlsym(handle, :llama_get_embeddings)
    global get_embeddings_ith_fn = dlsym(handle, :llama_get_embeddings_ith)

    # Model metadata functions
    global meta_val_str_fn = dlsym(handle, :llama_model_meta_val_str)
    global meta_count_fn = dlsym(handle, :llama_model_meta_count)
    global meta_key_fn = dlsym(handle, :llama_model_meta_key_by_index)
    global meta_val_fn = dlsym(handle, :llama_model_meta_val_str_by_index)

    # LoRA adapter functions
    global adapter_lora_init_fn = dlsym(handle, :llama_adapter_lora_init)
    global adapter_lora_free_fn = dlsym(handle, :llama_adapter_lora_free)
    global set_adapter_lora_fn = dlsym(handle, :llama_set_adapter_lora)
    global rm_adapter_lora_fn = dlsym(handle, :llama_rm_adapter_lora)
    global clear_adapter_lora_fn = dlsym(handle, :llama_clear_adapter_lora)

    println("✓ LlamaCpp module initialized successfully")
end

# ─────────────────────────────────────────────────────────────────────────────
# Model Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    model_default_params() -> llama_model_params

Get default model parameters.
"""
function model_default_params()
    return ccall(model_default_params_fn, llama_model_params, ())
end

"""
    load_model_from_file(path::String, params::llama_model_params) -> Ptr{Cvoid}

Load a model from a GGUF file.
"""
function load_model_from_file(path::String, params::llama_model_params)
    return ccall(model_load_fn, Ptr{Cvoid}, (Cstring, llama_model_params), path, params)
end

"""
    free_model(model::Ptr{Cvoid})

Free a loaded model.
"""
function free_model(model::Ptr{Cvoid})
    ccall(model_free_fn, Cvoid, (Ptr{Cvoid},), model)
end

"""
    get_vocab(model::Ptr{Cvoid}) -> Ptr{Cvoid}

Get the vocabulary from a model.
"""
function get_vocab(model::Ptr{Cvoid})
    return ccall(model_get_vocab_fn, Ptr{Cvoid}, (Ptr{Cvoid},), model)
end

"""
    model_desc(model::Ptr{Cvoid}, buf_size::Int=1024) -> String

Get model description string.
"""
function model_desc(model::Ptr{Cvoid}, buf_size::Integer=1024)
    buffer = Vector{UInt8}(undef, buf_size)
    result = ccall(model_desc_fn, Cint, (Ptr{Cvoid}, Ptr{UInt8}, Csize_t),
        model, pointer(buffer), buf_size)
    return String(copy(buffer[1:result]))
end

# ─────────────────────────────────────────────────────────────────────────────
# Context Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    context_default_params() -> llama_context_params

Get default context parameters.
"""
function context_default_params()
    return ccall(ctx_default_params_fn, llama_context_params, ())
end

"""
    new_context_with_model(model::Ptr{Cvoid}, params::llama_context_params) -> Ptr{Cvoid}

Create a new context with a model.
"""
function new_context_with_model(model::Ptr{Cvoid}, params::llama_context_params)
    return ccall(ctx_create_fn, Ptr{Cvoid}, (Ptr{Cvoid}, llama_context_params), model, params)
end

"""
    free_context(ctx::Ptr{Cvoid})

Free a context.
"""
function free_context(ctx::Ptr{Cvoid})
    ccall(ctx_free_fn, Cvoid, (Ptr{Cvoid},), ctx)
end

"""
    n_batch(ctx::Ptr{Cvoid}) -> UInt32

Get the maximum batch size for this context.
"""
function n_batch(ctx::Ptr{Cvoid})
    return ccall(n_batch_fn, UInt32, (Ptr{Cvoid},), ctx)
end

"""
    n_ctx(ctx::Ptr{Cvoid}) -> UInt32

Get the maximum context size for this context.
"""
function n_ctx(ctx::Ptr{Cvoid})
    return ccall(n_ctx_fn, UInt32, (Ptr{Cvoid},), ctx)
end

# ─────────────────────────────────────────────────────────────────────────────
# Tokenization Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    tokenize(vocab::Ptr{Cvoid}, text::String; add_special::Bool=true, parse_special::Bool=false) -> Vector{Cint}

Tokenize text into model tokens.
"""
function tokenize(vocab::Ptr{Cvoid}, text::String; add_special::Bool=true, parse_special::Bool=false)
    text_bytes = Vector{UInt8}(text)
    n = length(text_bytes)
    max_tokens = n + 32
    tokens = Vector{Cint}(undef, max_tokens)

    n_tokens = ccall(tokenize_fn, Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Cint, Ptr{Cint}, Cint, Bool, Bool),
        vocab, pointer(text_bytes), n, tokens, max_tokens, add_special, parse_special)

    return tokens[1:n_tokens]
end

"""
    vocab_type(vocab::Ptr{Cvoid}) -> LlamaVocabType

Get the tokenizer type used by this vocab.
"""
function vocab_type(vocab::Ptr{Cvoid})
    code = ccall(vocab_type_fn, Cint, (Ptr{Cvoid},), vocab)
    return LlamaVocabType(code)
end

"""
    vocab_n_tokens(vocab::Ptr{Cvoid}) -> Int

Get the vocabulary size.
"""
function vocab_n_tokens(vocab::Ptr{Cvoid})
    return Int(ccall(vocab_n_tokens_fn, Cint, (Ptr{Cvoid},), vocab))
end

"""
    token_get_text(vocab::Ptr{Cvoid}, token::Cint) -> String

Convert token ID to string.
"""
function vocab_get_text(vocab::Ptr{Cvoid}, token::Integer)
    cstr = ccall(vocab_get_text_fn, Cstring, (Ptr{Cvoid}, Cint), vocab, token)
    return unsafe_string(cstr)
end

"""
    token_eos(vocab::Ptr{Cvoid}) -> Cint

Get the end-of-sequence token ID.
"""
function token_eos(vocab::Ptr{Cvoid})
    return ccall(vocab_eos_fn, Cint, (Ptr{Cvoid},), vocab)
end

"""
    token_bos(vocab::Ptr{Cvoid}) -> Cint

Get the beginning-of-sequence token ID.
"""
function token_bos(vocab::Ptr{Cvoid})
    return ccall(vocab_bos_fn, Cint, (Ptr{Cvoid},), vocab)
end

"""
    token_is_eog(vocab::Ptr{Cvoid}, token::Cint) -> Bool

Check if token is end-of-generation.
"""
function token_is_eog(vocab::Ptr{Cvoid}, token::Cint)
    return ccall(vocab_is_eog_fn, Bool, (Ptr{Cvoid}, Cint), vocab, token)
end

# ─────────────────────────────────────────────────────────────────────────────
# Inference Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    decode(ctx::Ptr{Cvoid}, batch::llama_batch) -> Bool

Perform inference on a batch. Returns true on success.
"""
function decode(ctx::Ptr{Cvoid}, batch::llama_batch)
    ret = ccall(decode_fn, Cint, (Ptr{Cvoid}, llama_batch), ctx, batch)
    return ret == 0
end

"""
    get_logits(ctx::Ptr{Cvoid}, vocab_size::Int) -> Vector{Float32}

Get logits for the last token.
"""
function get_logits(ctx::Ptr{Cvoid}, vocab_size::Integer)
    ptr = ccall(get_logits_fn, Ptr{Cfloat}, (Ptr{Cvoid},), ctx)
    return unsafe_wrap(Array, ptr, (vocab_size,))
end

"""
    get_logits_ith(ctx::Ptr{Cvoid}, idx::Cint, vocab_size::Int) -> Vector{Float32}

Get logits for a specific token index.
"""
function get_logits_ith(ctx::Ptr{Cvoid}, idx::Cint, vocab_size::Integer)
    ptr = ccall(get_logits_ith_fn, Ptr{Cfloat}, (Ptr{Cvoid}, Cint), ctx, idx)
    if ptr == C_NULL
        return Float32[]
    end
    return unsafe_wrap(Array, ptr, (vocab_size,))
end

"""
    set_causal_attn(ctx::Ptr{Cvoid}, causal::Bool)

Set whether to use causal attention or not.
If set to true, the model will only attend to past tokens.
"""
function set_causal_attn(ctx::Ptr{Cvoid}, causal::Bool)
    ccall(set_causal_attn_fn, Cvoid, (Ptr{Cvoid}, Bool), ctx, causal)
end

# ─────────────────────────────────────────────────────────────────────────────
# Sampling Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    sampler_chain_default_params() -> llama_sampler_chain_params

Get default sampler chain parameters.
"""
function sampler_chain_default_params()
    return ccall(sampler_chain_default_params_fn, llama_sampler_chain_params, ())
end

"""
    sampler_chain_init(params::llama_sampler_chain_params) -> Ptr{Cvoid}

Initialize a sampler chain.
"""
function sampler_chain_init(params::llama_sampler_chain_params)
    return ccall(sampler_chain_init_fn, Ptr{Cvoid}, (llama_sampler_chain_params,), params)
end

"""
    sampler_chain_add(chain::Ptr{Cvoid}, sampler::Ptr{Cvoid})

Add a sampler to a sampler chain.
"""
function sampler_chain_add(chain::Ptr{Cvoid}, sampler::Ptr{Cvoid})
    ccall(sampler_chain_add_fn, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), chain, sampler)
end

"""
    sampler_init_top_k(k::Cint) -> Ptr{Cvoid}

Initialize a top-k sampler.
"""
function sampler_init_top_k(k::Integer)
    return ccall(sampler_init_topk_fn, Ptr{Cvoid}, (Cint,), k)
end

"""
    sampler_init_top_p(p::Float32, min_keep::Csize_t) -> Ptr{Cvoid}

Initialize a top-p (nucleus) sampler.
"""
function sampler_init_top_p(p::AbstractFloat, min_keep::Integer)
    return ccall(sampler_init_topp_fn, Ptr{Cvoid}, (Cfloat, Csize_t), p, min_keep)
end

"""
    sampler_init_temp(temp::Float32) -> Ptr{Cvoid}

Initialize a temperature sampler.
"""
function sampler_init_temp(temp::AbstractFloat)
    return ccall(sampler_init_temp_fn, Ptr{Cvoid}, (Cfloat,), temp)
end

"""
    sampler_init_penalties(last_n::Cint, repeat::Float32, freq::Float32, present::Float32) -> Ptr{Cvoid}

Initialize a penalties sampler.
"""
function sampler_init_penalties(last_n::Integer, repeat::AbstractFloat, freq::AbstractFloat, present::AbstractFloat)
    return ccall(sampler_init_penalties_fn, Ptr{Cvoid},
        (Cint, Cfloat, Cfloat, Cfloat), last_n, repeat, freq, present)
end

"""
    sampler_init_dist(seed::UInt32) -> Ptr{Cvoid}

Initialize a distribution sampler.
"""
function sampler_init_dist(seed::Integer)
    return ccall(sampler_init_dist_fn, Ptr{Cvoid}, (UInt32,), seed)
end

"""
    sampler_init_logit_bias(n_vocab::Cint, biases::Vector{LlamaLogitBias}) -> Ptr{Cvoid}

Initialize a logit bias sampler.
"""
function sampler_init_logit_bias(n_vocab::Cint, biases::Vector{LlamaLogitBias})
    n_pairs = Cint(length(biases))
    return ccall(sampler_init_logit_bias_fn, Ptr{Cvoid},
        (Cint, Cint, Ptr{LlamaLogitBias}), n_vocab, n_pairs, pointer(biases))
end

"""
    sampler_sample(sampler::Ptr{Cvoid}, ctx::Ptr{Cvoid}, idx::Cint) -> Cint

Sample a token using the sampler.
"""
function sampler_sample(sampler_chain::Ptr{Cvoid}, ctx::Ptr{Cvoid}, idx::Integer)
    return ccall(sampler_sample_fn, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Cint), sampler_chain, ctx, idx)
end

"""
    sampler_accept(sampler::Ptr{Cvoid}, token::Cint)

Accept a sampled token (updates sampler state).
"""
function sampler_accept(sampler_chain::Ptr{Cvoid}, token::Cint)
    ccall(sampler_accept_fn, Cvoid, (Ptr{Cvoid}, Cint), sampler_chain, token)
end

"""
    sampler_free(sampler::Ptr{Cvoid})

Free a sampler.
"""
function sampler_free(sampler_chain::Ptr{Cvoid})
    ccall(sampler_free_fn, Cvoid, (Ptr{Cvoid},), sampler_chain)
end

# ─────────────────────────────────────────────────────────────────────────────
# KV Cache Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    kv_cache_clear(ctx::Ptr{Cvoid})

Clear the entire KV cache.
"""
function kv_cache_clear(ctx::Ptr{Cvoid})
    ccall(kv_clear_fn, Cvoid, (Ptr{Cvoid},), ctx)
end

"""
    kv_cache_seq_rm(ctx::Ptr{Cvoid}, seq_id::Cint, p0::Cint, p1::Cint) -> Bool

Remove cached tokens from sequence in position range [p0, p1).
"""
function kv_cache_seq_rm(ctx::Ptr{Cvoid}, seq_id::Cint, p0::Cint, p1::Cint)
    return ccall(kv_seq_rm_fn, Bool, (Ptr{Cvoid}, Cint, Cint, Cint), ctx, seq_id, p0, p1)
end

"""
    kv_cache_seq_keep(ctx::Ptr{Cvoid}, seq_id::Cint)

Keep only the specified sequence in KV cache.
"""
function kv_cache_seq_keep(ctx::Ptr{Cvoid}, seq_id::Cint)
    ccall(kv_seq_keep_fn, Cvoid, (Ptr{Cvoid}, Cint), ctx, seq_id)
end

"""
    kv_cache_seq_cp(ctx::Ptr{Cvoid}, seq_id_src::Cint, seq_id_dst::Cint, p0::Cint, p1::Cint)

Copy sequence data from src to dst in position range [p0, p1).
"""
function kv_cache_seq_cp(ctx::Ptr{Cvoid}, seq_id_src::Cint, seq_id_dst::Cint, p0::Cint, p1::Cint)
    ccall(kv_seq_cp_fn, Cvoid, (Ptr{Cvoid}, Cint, Cint, Cint, Cint),
        ctx, seq_id_src, seq_id_dst, p0, p1)
end

"""
    kv_cache_seq_add(ctx::Ptr{Cvoid}, seq_id::Cint, p0::Cint, p1::Cint, delta::Cint)

Add delta to RoPE positions in sequence for range [p0, p1).
"""
function kv_cache_seq_add(ctx::Ptr{Cvoid}, seq_id::Cint, p0::Cint, p1::Cint, delta::Cint)
    ccall(kv_seq_add_fn, Cvoid, (Ptr{Cvoid}, Cint, Cint, Cint, Cint),
        ctx, seq_id, p0, p1, delta)
end

"""
    get_kv_cache_used_cells(ctx::Ptr{Cvoid}) -> Int

Get the number of tokens currently in KV cache.
"""
function get_kv_cache_used_cells(ctx::Ptr{Cvoid})
    return Int(ccall(kv_self_n_tokens_fn, Cint, (Ptr{Cvoid},), ctx))
end

# ─────────────────────────────────────────────────────────────────────────────
# Embedding Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    set_embeddings(ctx::Ptr{Cvoid}, embeddings::Bool)

Enable/disable embeddings mode.
"""
function set_embeddings(ctx::Ptr{Cvoid}, embeddings::Bool)
    ccall(set_embeddings_fn, Cvoid, (Ptr{Cvoid}, Bool), ctx, embeddings)
end

"""
    get_embeddings(ctx::Ptr{Cvoid}, n_outputs::Int, n_embd::Int) -> Array{Float32,2}

Get embeddings matrix after decode in embeddings mode.
"""
function get_embeddings(ctx::Ptr{Cvoid}, n_outputs::Int, n_embd::Int)
    ptr = ccall(get_embeddings_fn, Ptr{Cfloat}, (Ptr{Cvoid},), ctx)
    if ptr == C_NULL
        return Array{Float32,2}(undef, 0, 0)
    end
    flat = unsafe_wrap(Array, ptr, (n_outputs * n_embd,))
    return reshape(flat, (n_embd, n_outputs))'
end

"""
    get_embeddings_ith(ctx::Ptr{Cvoid}, idx::Cint, n_embd::Int) -> Vector{Float32}

Get embedding vector for specific token index.
"""
function get_embeddings_ith(ctx::Ptr{Cvoid}, idx::Cint, n_embd::Int)
    ptr = ccall(get_embeddings_ith_fn, Ptr{Cfloat}, (Ptr{Cvoid}, Cint), ctx, idx)
    if ptr == C_NULL
        return Float32[]
    end
    return unsafe_wrap(Array, ptr, (n_embd,))
end

# ─────────────────────────────────────────────────────────────────────────────
# Model Metadata Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    model_meta_val_str(model::Ptr{Cvoid}, key::String; bufsize::Int=1024) -> Union{String, Nothing}

Get metadata value for a specific key.
"""
function model_meta_val_str(model::Ptr{Cvoid}, key::String; bufsize::Int=1024)
    buf = Vector{UInt8}(undef, bufsize)
    ret = ccall(meta_val_str_fn, Cint,
        (Ptr{Cvoid}, Cstring, Ptr{UInt8}, Csize_t),
        model, key, pointer(buf), bufsize)

    if ret < 0
        return nothing
    end
    return String(buf[1:ret])
end

"""
    model_meta_count(model::Ptr{Cvoid}) -> Int

Get the number of metadata entries.
"""
function model_meta_count(model::Ptr{Cvoid})
    return ccall(meta_count_fn, Cint, (Ptr{Cvoid},), model)
end

"""
    model_meta_key_by_index(model::Ptr{Cvoid}, idx::Int; bufsize::Int=256) -> Union{String, Nothing}

Get metadata key by index.
"""
function model_meta_key_by_index(model::Ptr{Cvoid}, idx::Int; bufsize::Int=256)
    buf = Vector{UInt8}(undef, bufsize)
    ret = ccall(meta_key_fn, Cint,
        (Ptr{Cvoid}, Cint, Ptr{UInt8}, Csize_t),
        model, idx, pointer(buf), bufsize)
    ret < 0 && return nothing
    return String(buf[1:ret])
end

"""
    model_meta_val_by_index(model::Ptr{Cvoid}, idx::Int; bufsize::Int=10000) -> Union{String, Nothing}

Get metadata value by index.
"""
function model_meta_val_by_index(model::Ptr{Cvoid}, idx::Int; bufsize::Int=10000)
    buf = Vector{UInt8}(undef, bufsize)
    ret = ccall(meta_val_fn, Cint,
        (Ptr{Cvoid}, Cint, Ptr{UInt8}, Csize_t),
        model, idx, pointer(buf), bufsize)
    ret < 0 && return nothing
    return String(buf[1:ret])
end

"""
    list_model_metadata(model::Ptr{Cvoid}) -> Dict{String, String}

Get all model metadata as a dictionary.
"""
function list_model_metadata(model::Ptr{Cvoid})
    n = model_meta_count(model)
    metadata = Dict{String,String}()
    for i in 0:(n-1)
        key = model_meta_key_by_index(model, i)
        val = key === nothing ? nothing : model_meta_val_by_index(model, i)
        if key !== nothing && val !== nothing
            metadata[key] = val
        end
    end
    return metadata
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions for Batch Construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_batch(tokens::Vector{Cint}) -> Tuple{llama_batch, Tuple}

Build a llama_batch from tokens with keep-alive references.
"""
function build_batch(tokens::Vector{Cint})
    n = length(tokens)
    pos_data = collect(Cint(i - 1) for i in 1:n)
    n_seq_id_data = fill(Cint(1), n)
    seq_id_inner = [Cint[0] for _ in 1:n]
    seq_id_ptrs = [pointer(seq_id_inner[i]) for i in 1:n]
    logits_data = fill(Int8(0), n)
    logits_data[end] = Int8(1)  # Only compute logits for last token

    batch = llama_batch(
        Cint(n),
        pointer(tokens),
        Ptr{Cfloat}(C_NULL),
        pointer(pos_data),
        pointer(n_seq_id_data),
        pointer(seq_id_ptrs),
        pointer(logits_data)
    )

    keep_alive = (tokens, pos_data, n_seq_id_data, seq_id_inner, seq_id_ptrs, logits_data)
    return batch, keep_alive
end

"""
    build_single_token_batch(token_id::Cint, pos::Cint; seq_id::Cint=Cint(0)) -> Tuple{llama_batch, Tuple}

Build a llama_batch for a single token with keep-alive references.
"""
function build_single_token_batch(token_id::Cint, pos::Cint; seq_id::Cint=Cint(0))
    tokens_vec = [token_id]
    pos_vec = [pos]
    nseq_vec = [Cint(1)]
    seq_id_inner = [seq_id]
    seq_id_ptrs = [pointer(seq_id_inner)]
    logits_vec = [Int8(1)]

    batch = llama_batch(
        Cint(1),
        pointer(tokens_vec),
        Ptr{Cfloat}(C_NULL),
        pointer(pos_vec),
        pointer(nseq_vec),
        pointer(seq_id_ptrs),
        pointer(logits_vec)
    )

    keep_alive = (tokens_vec, pos_vec, nseq_vec, seq_id_inner, seq_id_ptrs, logits_vec)
    return batch, keep_alive
end

# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    clean_tokenizer_artifacts(text::String) -> String

Clean common tokenizer artifacts from decoded text.
"""
function clean_tokenizer_artifacts(text::String)
    cleaned = replace(text, "Ġ" => " ")    # SentencePiece space token
    cleaned = replace(cleaned, "▁" => " ") # Another common space token
    cleaned = replace(cleaned, "Ċ" => "\n") # Newline token
    cleaned = replace(cleaned, "ĉ" => "\t") # Tab token
    cleaned = replace(cleaned, "Â" => "")   # Sometimes appears with spaces
    return cleaned
end

"""
    token_to_string_clean(model::Ptr{Cvoid}, token::Cint) -> String

Convert token to string and clean tokenizer artifacts.
"""
function token_to_string_clean(model::Ptr{Cvoid}, token::Cint)
    raw_str = token_get_text(model, token)
    return clean_tokenizer_artifacts(raw_str)
end


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Adapter Functions (add these new functions)
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_lora_adapter(model::Ptr{Cvoid}, path_lora::String) -> Ptr{Cvoid}

Load a LoRA adapter from file.

# Arguments
- `model::Ptr{Cvoid}`: Pointer to the loaded base model
- `path_lora::String`: Path to the LoRA adapter file (.gguf format)

# Returns
- `Ptr{Cvoid}`: Pointer to the loaded LoRA adapter, or C_NULL on failure

# Example
```julia
adapter = load_lora_adapter(model, "path/to/economics_lora.gguf")
```
"""
function load_lora_adapter(model::Ptr{Cvoid}, path_lora::String)
    return ccall(adapter_lora_init_fn, Ptr{Cvoid},
        (Ptr{Cvoid}, Cstring), model, path_lora)
end

"""
    free_lora_adapter(adapter::Ptr{Cvoid})

Manually free a LoRA adapter.
Note: Loaded adapters will be automatically freed when the associated model is deleted.

# Arguments
- `adapter::Ptr{Cvoid}`: Pointer to the LoRA adapter to free
"""
function free_lora_adapter(adapter::Ptr{Cvoid})
    ccall(adapter_lora_free_fn, Cvoid, (Ptr{Cvoid},), adapter)
end

"""
    set_adapter_lora(ctx::Ptr{Cvoid}, adapter::Ptr{Cvoid}, scale::Float32=1.0f0) -> Int32

Add a loaded LoRA adapter to the given context.
This will not modify the model's weights permanently.

# Arguments
- `ctx::Ptr{Cvoid}`: Pointer to the llama context
- `adapter::Ptr{Cvoid}`: Pointer to the loaded LoRA adapter
- `scale::Float32`: Scaling factor for the adapter (default: 1.0)

# Returns
- `Int32`: 0 on success, negative on failure

# Example
```julia
result = set_adapter_lora(ctx, economics_adapter, 1.0f0)
if result == 0
    println("✓ LoRA adapter applied successfully")
else
    println("✗ Failed to apply LoRA adapter")
end
```
"""
function set_adapter_lora(ctx::Ptr{Cvoid}, adapter::Ptr{Cvoid}, scale::AbstractFloat=1.0f0)
    return ccall(set_adapter_lora_fn, Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Cfloat), ctx, adapter, scale)
end

"""
    rm_adapter_lora(ctx::Ptr{Cvoid}, adapter::Ptr{Cvoid}) -> Int32

Remove a specific LoRA adapter from the given context.

# Arguments
- `ctx::Ptr{Cvoid}`: Pointer to the llama context
- `adapter::Ptr{Cvoid}`: Pointer to the LoRA adapter to remove

# Returns
- `Int32`: 0 on success, -1 if adapter not present in context

# Example
```julia
result = rm_adapter_lora(ctx, economics_adapter)
if result == 0
    println("✓ LoRA adapter removed successfully")
else
    println("✗ LoRA adapter not found in context")
end
```
"""
function rm_adapter_lora(ctx::Ptr{Cvoid}, adapter::Ptr{Cvoid})
    return ccall(rm_adapter_lora_fn, Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}), ctx, adapter)
end

"""
    clear_adapter_lora(ctx::Ptr{Cvoid})

Remove all LoRA adapters from the given context.

# Arguments
- `ctx::Ptr{Cvoid}`: Pointer to the llama context

# Example
```julia
clear_adapter_lora(ctx)
println("✓ All LoRA adapters cleared from context")
```
"""
function clear_adapter_lora(ctx::Ptr{Cvoid})
    ccall(clear_adapter_lora_fn, Cvoid, (Ptr{Cvoid},), ctx)
end

# ─────────────────────────────────────────────────────────────────────────────
# High-Level LoRA Management Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    LoRAManager

A struct to manage multiple LoRA adapters and their lifecycle.
"""
mutable struct LoRAManager
    model::Ptr{Cvoid}
    adapters::Dict{String,Ptr{Cvoid}}
    active_adapter::Union{String,Nothing}

    function LoRAManager(model::Ptr{Cvoid})
        new(model, Dict{String,Ptr{Cvoid}}(), nothing)
    end
end

"""
    load_adapter!(manager::LoRAManager, name::String, path::String) -> Bool

Load a LoRA adapter and register it with the manager.

# Arguments
- `manager::LoRAManager`: The LoRA manager instance
- `name::String`: Identifier for the adapter (e.g., "economics_v1")
- `path::String`: Path to the LoRA adapter file

# Returns
- `Bool`: true on success, false on failure
"""
function load_adapter!(manager::LoRAManager, name::String, path::String)
    if haskey(manager.adapters, name)
        @warn "Adapter '$name' already loaded, skipping"
        return true
    end

    adapter = load_lora_adapter(manager.model, path)
    if adapter == C_NULL
        @error "Failed to load LoRA adapter from '$path'"
        return false
    end

    manager.adapters[name] = adapter
    @info "✓ Loaded LoRA adapter '$name' from '$path'"
    return true
end

"""
    switch_adapter!(manager::LoRAManager, ctx::Ptr{Cvoid}, name::String; scale::Float32=1.0f0) -> Bool

Switch to a different LoRA adapter in the given context.

# Arguments
- `manager::LoRAManager`: The LoRA manager instance
- `ctx::Ptr{Cvoid}`: Pointer to the llama context
- `name::String`: Name of the adapter to switch to
- `scale::Float32`: Scaling factor for the adapter

# Returns
- `Bool`: true on success, false on failure
"""
function switch_adapter!(manager::LoRAManager, ctx::Ptr{Cvoid}, name::String; scale::AbstractFloat=1.0f0)
    if !haskey(manager.adapters, name)
        @error "Adapter '$name' not found. Available adapters: $(keys(manager.adapters))"
        return false
    end

    # Clear any existing adapters
    clear_adapter_lora(ctx)

    # Apply the requested adapter
    adapter = manager.adapters[name]
    result = set_adapter_lora(ctx, adapter, scale)

    if result == 0
        manager.active_adapter = name
        @info "✓ Switched to LoRA adapter '$name' with scale $scale"
        return true
    else
        @error "Failed to apply LoRA adapter '$name'"
        return false
    end
end

"""
    clear_active_adapter!(manager::LoRAManager, ctx::Ptr{Cvoid})

Clear the currently active adapter from the context.
"""
function clear_active_adapter!(manager::LoRAManager, ctx::Ptr{Cvoid})
    clear_adapter_lora(ctx)
    manager.active_adapter = nothing
    @info "✓ Cleared active LoRA adapter"
end

"""
    free_all_adapters!(manager::LoRAManager)

Free all loaded adapters. Call this before freeing the model.
"""
function free_all_adapters!(manager::LoRAManager)
    for (name, adapter) in manager.adapters
        free_lora_adapter(adapter)
        @info "✓ Freed LoRA adapter '$name'"
    end
    empty!(manager.adapters)
    manager.active_adapter = nothing
end

"""
    list_adapters(manager::LoRAManager) -> Vector{String}

Get a list of all loaded adapter names.
"""
function list_adapters(manager::LoRAManager)
    return collect(keys(manager.adapters))
end

"""
    get_active_adapter(manager::LoRAManager) -> Union{String, Nothing}

Get the name of the currently active adapter, or nothing if none is active.
"""
function get_active_adapter(manager::LoRAManager)
    return manager.active_adapter
end

# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

export LlamaVocabType
export llama_model_params, llama_context_params, llama_sampler_chain_params
export llama_batch, LlamaLogitBias

# Initialization
export init!

# Model functions
export model_default_params, load_model_from_file, free_model, get_vocab, model_desc

# Context functions  
export context_default_params, new_context_with_model, free_context, n_batch, n_ctx

# Tokenization functions
export tokenize, vocab_type, vocab_n_tokens, vocab_get_text, token_eos, token_bos, token_is_eog

# Inference functions
export decode, get_logits, get_logits_ith

# Sampling functions
export sampler_chain_default_params, sampler_chain_init, sampler_chain_add
export sampler_init_top_k, sampler_init_top_p, sampler_init_temp, sampler_init_penalties
export sampler_init_dist, sampler_init_logit_bias
export sampler_sample, sampler_accept, sampler_free

# KV cache functions
export kv_cache_clear, kv_cache_seq_rm, kv_cache_seq_keep, kv_cache_seq_cp
export kv_cache_seq_add, get_kv_cache_used_cells

# Embedding functions
export set_embeddings, get_embeddings, get_embeddings_ith

# Metadata functions
export model_meta_val_str, model_meta_count, model_meta_key_by_index
export model_meta_val_by_index, list_model_metadata

# Helper functions
export build_batch, build_single_token_batch
export clean_tokenizer_artifacts, token_to_string_clean

# LoRA adapter functions
export load_lora_adapter, free_lora_adapter, set_adapter_lora, rm_adapter_lora, clear_adapter_lora
export LoRAManager, load_adapter!, switch_adapter!, clear_active_adapter!, free_all_adapters!
export list_adapters, get_active_adapter

end # module LlamaCpp