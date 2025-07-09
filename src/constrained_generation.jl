# ─────────────────────────────────────────────────────────────────────────────
# Constrained Generation Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    greedy_constrained_generation(prompt::String, schema::Dict, tokenizer::String; 
                                 max_tokens::Int=50, 
                                 hf_token::String="",
                                 model_context, 
                                 vocab,
                                 verbose::Bool=true) -> String

Generate text using greedy sampling with Outlines constraints.

# Arguments
- `prompt::String`: Input prompt for generation
- `schema::Dict`: JSON schema defining the output structure
- `tokenizer::String`: HuggingFace model name for tokenizer (e.g., "google/gemma-3-4b-it")

# Keyword Arguments
- `max_tokens::Int=50`: Maximum number of tokens to generate
- `hf_token::String=""`: HuggingFace authentication token (if needed)
- `model_context`: Pre-loaded model context
- `vocab`: Pre-loaded vocab
- `verbose::Bool=true`: Print generation progress

# Returns
- `String`: Raw generated text without cleaning (includes BPE tokens like ▁, Ġ)

# Example
```julia
schema = Dict(
    "type" => "object",
    "properties" => Dict("city" => Dict("type" => "string")),
    "required" => ["city"]
)

result = greedy_constrained_generation(
    "What is the capital of France?", 
    schema, 
    "google/gemma-3-4b-it",
    max_tokens=30,
    model_context=model_context,
    vocab=vocab
)
```
"""
function greedy_constrained_generation(prompt::String, schema::Dict, tokenizer::String;
    max_tokens::Int=50,
    hf_token::String="",
    model_context,
    vocab,
    verbose::Bool=true)

    if verbose
        println("🚀 Starting greedy constrained generation...")
        println("   Tokenizer: $tokenizer")
        println("   Max tokens: $max_tokens")
    end

    # Step 1: Setup Outlines constraints
    if verbose
        println("🔧 Setting up Outlines constraints...")
    end

    # Create vocabulary handle
    if isempty(hf_token)
        outlines_vocab = outlines.create_vocabulary(tokenizer)
    else
        outlines_vocab = outlines.create_vocabulary_with_token(tokenizer, hf_token)
    end

    # Generate regex from schema
    schema_json = JSON3.write(schema)
    regex = outlines.regex_from_schema(schema_json)

    # Create index
    index_handle = outlines.create_index(regex, outlines_vocab)

    if verbose
        println("✅ Outlines setup complete")
    end

    try
        # Step 2: Process prompt
        if verbose
            println("🔧 Processing prompt...")
        end

        # Clear context
        llama.kv_cache_clear(model_context)
        llama.set_embeddings(model_context, false)

        # Tokenize and decode prompt
        prompt_tokens = llama.tokenize(vocab, prompt; add_special=true, parse_special=false)
        batch, keep = llama.build_batch(prompt_tokens)
        success = llama.decode(model_context, batch)

        if !success
            error("Failed to decode prompt")
        end

        if verbose
            println("✅ Prompt processed ($(length(prompt_tokens)) tokens)")
        end

        # Step 3: Initialize generation state
        current_state = outlines.index_initial_state(index_handle)
        current_pos = length(prompt_tokens)
        generated_text = ""

        # Get vocab size locally
        vocab_size = llama.vocab_n_tokens(vocab)

        if verbose
            println("🎯 Starting generation loop...")
            println("   Initial state: $current_state")
            println("   Vocab size: $vocab_size")
        end

        # Step 4: Generation loop
        for step in 1:max_tokens
            if verbose
                println("\n🔒 Step $step:")
                println("   State: $current_state")
                println("   Generated: '$generated_text'")
            end

            # Get allowed tokens from Outlines
            allowed_tokens = outlines.index_allowed_tokens(index_handle, current_state)

            if length(allowed_tokens) == 0
                if verbose
                    println("❌ No allowed tokens")
                end
                break
            end

            if verbose
                println("   Allowed: $(length(allowed_tokens)) tokens")
            end

            # Get logits and apply constraints
            logits = llama.get_logits(model_context, vocab_size)

            # Apply constraint masking
            masked_logits = fill(-Inf32, vocab_size)
            allowed_count = 0
            for token_id in allowed_tokens
                if token_id < vocab_size
                    masked_logits[token_id+1] = logits[token_id+1]  # Julia 1-indexed
                    allowed_count += 1
                end
            end
            logits .= masked_logits

            # Check for valid tokens
            valid_indices = findall(x -> x != -Inf32, logits)
            if isempty(valid_indices)
                if verbose
                    println("❌ No valid tokens after masking")
                end
                break
            end

            # Greedy sampling (argmax)
            max_idx = argmax(logits)
            next_token = UInt32(max_idx - 1)  # Convert to 0-indexed

            # Get token text and update generated string
            token_text = llama.vocab_get_text(vocab, Cint(next_token))
            generated_text *= token_text

            if verbose
                println("   🎲 Sampled: token $next_token -> '$token_text'")
            end

            # Update Outlines state
            next_state = outlines.index_next_state(index_handle, current_state, next_token)
            current_state = next_state

            # Check if final state
            is_final = outlines.index_is_final_state(index_handle, current_state)
            if is_final
                if verbose
                    println("🎉 Final state reached!")
                end
                break
            end

            # Update model context
            batch, keep = llama.build_single_token_batch(Cint(next_token), Cint(current_pos))
            success = llama.decode(model_context, batch)
            if !success
                if verbose
                    println("❌ Failed to decode token")
                end
                break
            end

            current_pos += 1
        end

        if verbose
            println("\n✅ Generation complete!")
            println("🏁 Raw result: '$generated_text'")
        end

        return generated_text

    finally
        # Always cleanup Outlines resources
        outlines.free_index(index_handle)
        outlines.free_vocabulary(outlines_vocab)
        if verbose
            println("🧹 Cleaned up Outlines resources")
        end
    end
end

"""
    greedy_mtmd_constrained_generation(prompt::String, img_paths::Vector{String}, schema::Dict, tokenizer::String; 
                                      max_tokens::Int=50, 
                                      hf_token::String="",
                                      model,
                                      model_context, 
                                      vocab,
                                      mtmd_context,
                                      verbose::Bool=true) -> String

Generate multimodal text using greedy sampling with Outlines constraints.

# Arguments
- `prompt::String`: Input prompt for generation
- `img_paths::Vector{String}`: Paths to image files
- `schema::Dict`: JSON schema defining the output structure
- `tokenizer::String`: HuggingFace model name for tokenizer (e.g., "google/gemma-3-4b-it")

# Keyword Arguments
- `max_tokens::Int=50`: Maximum number of tokens to generate
- `hf_token::String=""`: HuggingFace authentication token (if needed)
- `model`: Pre-loaded model
- `model_context`: Pre-loaded model context
- `vocab`: Pre-loaded vocab
- `mtmd_context`: Pre-loaded MTMD context
- `verbose::Bool=true`: Print generation progress

# Returns
- `String`: Raw generated text without cleaning (includes BPE tokens like ▁, Ġ)

# Example
```julia
schema = Dict(
    "type" => "object",
    "properties" => Dict(
        "caption" => Dict("type" => "string"),
        "objects" => Dict("type" => "array", "items" => Dict("type" => "string"))
    ),
    "required" => ["caption", "objects"]
)

result = greedy_mtmd_constrained_generation(
    "Describe this image in JSON format:", 
    ["path/to/image.jpg"],
    schema, 
    "google/gemma-3-4b-it",
    model=model,
    model_context=model_context,
    vocab=vocab,
    mtmd_context=mtmd_context
)
```
"""
function greedy_mtmd_constrained_generation(prompt::String, img_paths::Vector{String}, schema::Dict, tokenizer::String;
    max_tokens::Int=50,
    hf_token::String="",
    model,
    model_context,
    vocab,
    mtmd_context,
    verbose::Bool=true)

    if verbose
        println("🚀 Starting multimodal greedy constrained generation...")
        println("   Tokenizer: $tokenizer")
        println("   Images: $(length(img_paths))")
        println("   Max tokens: $max_tokens")
    end

    # Step 1: Setup Outlines constraints
    if verbose
        println("🔧 Setting up Outlines constraints...")
    end

    # Create vocabulary handle
    if isempty(hf_token)
        outlines_vocab = outlines.create_vocabulary(tokenizer)
    else
        outlines_vocab = outlines.create_vocabulary_with_token(tokenizer, hf_token)
    end

    # Generate regex from schema
    schema_json = JSON3.write(schema)
    regex = outlines.regex_from_schema(schema_json)

    # Create index
    index_handle = outlines.create_index(regex, outlines_vocab)

    if verbose
        println("✅ Outlines setup complete")
    end

    try
        # Step 2: Multimodal encoding
        if verbose
            println("🔧 Processing multimodal input...")
        end

        # Clear KV cache and disable embeddings
        llama.kv_cache_clear(model_context)
        llama.set_embeddings(model_context, false)

        # Load image bitmaps
        image_bitmaps = [load_an_image_bitmap(path) for path in img_paths]
        if verbose
            println("📷 Loaded $(length(image_bitmaps)) images")
        end

        # Initialize chunks
        chunks = mtmd.input_chunks_init()

        # Create input text
        prompt_text = mtmd.create_input_text(prompt)

        # Tokenize multimodal input
        ret = mtmd.tokenize(mtmd_context, chunks, prompt_text, image_bitmaps)
        if ret != 0
            error("Multimodal tokenization failed with code $ret")
        end

        if verbose
            println("✅ Multimodal tokenization complete")
        end

        # Evaluate chunks to get initial context
        n_past = 0
        seq_id = 0
        n_batch = llama.n_batch(model_context)

        success, new_n_past = mtmd.helper_eval_chunks(mtmd_context, model_context, chunks,
            n_past, seq_id, n_batch, true)

        if !success
            error("Failed to evaluate multimodal chunks")
        end

        if verbose
            println("✅ Multimodal context processed")
            println("   Context length: $new_n_past tokens")
        end

        # Step 3: Initialize generation state
        current_state = outlines.index_initial_state(index_handle)
        current_pos = new_n_past  # Start from end of multimodal context
        generated_text = ""

        # Get vocab size locally
        vocab_size = llama.vocab_n_tokens(vocab)

        if verbose
            println("🎯 Starting constrained generation loop...")
            println("   Initial state: $current_state")
            println("   Starting position: $current_pos")
            println("   Vocab size: $vocab_size")
        end

        # Step 4: Constrained generation loop
        for step in 1:max_tokens
            if verbose
                println("\n🔒 Step $step:")
                println("   State: $current_state")
                println("   Generated: '$generated_text'")
            end

            # Get allowed tokens from Outlines
            allowed_tokens = outlines.index_allowed_tokens(index_handle, current_state)

            if length(allowed_tokens) == 0
                if verbose
                    println("❌ No allowed tokens")
                end
                break
            end

            if verbose
                println("   Allowed: $(length(allowed_tokens)) tokens")
            end

            # Get logits and apply constraints
            logits = llama.get_logits(model_context, vocab_size)

            # Apply constraint masking
            masked_logits = fill(-Inf32, vocab_size)
            allowed_count = 0
            for token_id in allowed_tokens
                if token_id < vocab_size
                    masked_logits[token_id+1] = logits[token_id+1]  # Julia 1-indexed
                    allowed_count += 1
                end
            end
            logits .= masked_logits

            # Check for valid tokens
            valid_indices = findall(x -> x != -Inf32, logits)
            if isempty(valid_indices)
                if verbose
                    println("❌ No valid tokens after masking")
                end
                break
            end

            # Greedy sampling (argmax)
            max_idx = argmax(logits)
            next_token = UInt32(max_idx - 1)  # Convert to 0-indexed

            # Get token text and update generated string
            token_text = llama.vocab_get_text(vocab, Cint(next_token))
            generated_text *= token_text

            if verbose
                println("   🎲 Sampled: token $next_token -> '$token_text'")
            end

            # Update Outlines state
            next_state = outlines.index_next_state(index_handle, current_state, next_token)
            current_state = next_state

            # Check if final state
            is_final = outlines.index_is_final_state(index_handle, current_state)
            if is_final
                if verbose
                    println("🎉 Final state reached!")
                end
                break
            end

            # Update model context
            batch, keep = llama.build_single_token_batch(Cint(next_token), Cint(current_pos))
            success = llama.decode(model_context, batch)
            if !success
                if verbose
                    println("❌ Failed to decode token")
                end
                break
            end

            current_pos += 1
        end

        if verbose
            println("\n✅ Multimodal generation complete!")
            println("🏁 Raw result: '$generated_text'")
        end

        # Cleanup multimodal resources
        mtmd.input_chunks_free(chunks)
        for bitmap in image_bitmaps
            mtmd.bitmap_free(bitmap)
        end

        return generated_text

    finally
        # Always cleanup Outlines resources
        outlines.free_index(index_handle)
        outlines.free_vocabulary(outlines_vocab)
        if verbose
            println("🧹 Cleaned up Outlines resources")
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Sampling Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
Apply temperature scaling to logits.
"""
function apply_temperature!(logits::Vector{Float32}, temperature::AbstractFloat)
    if temperature <= 0.0f0
        # Greedy: set max to very high, others to very low
        max_idx = argmax(logits)
        fill!(logits, -Inf32)
        logits[max_idx] = 1000.0f0
    elseif temperature != 1.0f0
        logits ./= temperature
    end
    return logits
end

"""
Apply top-k filtering to logits.
"""
function apply_top_k!(logits::Vector{Float32}, k::Integer)
    if k <= 0 || k >= length(logits)
        return logits
    end

    # Get indices of top-k values
    indices = sortperm(logits, rev=true)[1:k]

    # Mask everything else
    masked_logits = fill(-Inf32, length(logits))
    for idx in indices
        masked_logits[idx] = logits[idx]
    end

    return masked_logits
end

"""
Apply top-p (nucleus) sampling to logits.
"""
function apply_top_p!(logits::Vector{Float32}, p::AbstractFloat)
    if p >= 1.0f0
        return logits
    end

    # Convert to probabilities
    probs = softmax(logits)

    # Sort by probability (descending)
    sorted_indices = sortperm(probs, rev=true)

    # Find cutoff point
    cumsum_probs = 0.0f0
    cutoff_idx = length(probs)

    for (i, idx) in enumerate(sorted_indices)
        cumsum_probs += probs[idx]
        if cumsum_probs >= p
            cutoff_idx = i
            break
        end
    end

    # Mask tokens beyond cutoff
    masked_logits = fill(-Inf32, length(logits))
    for i in 1:cutoff_idx
        idx = sorted_indices[i]
        masked_logits[idx] = logits[idx]
    end

    return masked_logits
end

"""
Apply min-p sampling to logits.
"""
function apply_min_p!(logits::Vector{Float32}, min_p::AbstractFloat)
    if min_p <= 0.0f0
        return logits
    end

    max_logit = maximum(logits[isfinite.(logits)])
    threshold = max_logit + log(min_p)

    # Mask tokens below threshold
    for i in eachindex(logits)
        if isfinite(logits[i]) && logits[i] < threshold
            logits[i] = -Inf32
        end
    end

    return logits
end

"""
Apply repetition penalty to logits based on recent tokens.
"""
function apply_repetition_penalty!(logits::Vector{Float32}, recent_tokens::Vector{Int32},
    penalty::AbstractFloat, vocab_size::Int32)
    if penalty == 1.0f0 || isempty(recent_tokens)
        return logits
    end

    for token_id in recent_tokens
        if token_id >= 0 && token_id < vocab_size
            julia_idx = token_id + 1  # Convert to 1-indexed
            if julia_idx <= length(logits)
                if logits[julia_idx] > 0
                    logits[julia_idx] /= penalty
                else
                    logits[julia_idx] *= penalty
                end
            end
        end
    end

    return logits
end

"""
Softmax function for probability conversion.
"""
function softmax(logits::Vector{Float32})
    # Handle infinite values
    finite_mask = isfinite.(logits)
    if !any(finite_mask)
        # All infinite, return uniform over finite values
        return fill(1.0f0 / length(logits), length(logits))
    end

    max_logit = maximum(logits[finite_mask])
    exp_logits = zeros(Float32, length(logits))

    for i in eachindex(logits)
        if finite_mask[i]
            exp_logits[i] = exp(logits[i] - max_logit)
        end
    end

    sum_exp = sum(exp_logits)
    return exp_logits ./ sum_exp
end

"""
Sample a token from processed logits.
"""
function sample_token_enhanced(logits::Vector{Float32}, rng::AbstractRNG)
    # Check for valid tokens
    valid_indices = findall(x -> isfinite(x), logits)
    if isempty(valid_indices)
        error("No valid tokens to sample from")
    end

    # If only one valid token, return it
    if length(valid_indices) == 1
        return UInt32(valid_indices[1] - 1)  # Convert to 0-indexed
    end

    # Convert to probabilities and sample
    probs = softmax(logits)

    # Multinomial sampling
    r = rand(rng)
    cumsum_prob = 0.0f0

    for (i, prob) in enumerate(probs)
        cumsum_prob += prob
        if r <= cumsum_prob
            return UInt32(i - 1)  # Convert to 0-indexed
        end
    end

    # Fallback: return last valid token
    return UInt32(valid_indices[end] - 1)
end

"""
    enhanced_constrained_generation(prompt::String, schema::Dict, tokenizer::String; max_tokens::Int=50, hf_token::String="", model_context, vocab, sampling_params::SamplingParams=balanced_params(), verbose::Bool=true)

Enhanced constrained generation with sampling parameters.

# Arguments
- `prompt::String`: Input prompt for generation
- `schema::Dict`: JSON schema defining the output structure
- `tokenizer::String`: HuggingFace model name for tokenizer

# Keyword Arguments
- `max_tokens::Int=50`: Maximum number of tokens to generate
- `hf_token::String=""`: HuggingFace authentication token (if needed)
- `model_context`: Pre-loaded model context
- `vocab`: Pre-loaded vocab
- `sampling_params::SamplingParams=balanced_params()`: Sampling configuration
- `verbose::Bool=true`: Print generation progress

# Returns
- `String`: Generated text

# Example
```julia
result = enhanced_constrained_generation(
    "What is the capital of France?",
    schema,
    "google/gemma-3-4b-it",
    model_context=model_context,
    vocab=vocab,
    sampling_params=creative_params()
)
```
"""
function enhanced_constrained_generation(prompt::String, schema::Dict, tokenizer::String;
    max_tokens::Int=50,
    hf_token::String="",
    model_context,
    vocab,
    sampling_params::SamplingParams=balanced_params(),
    verbose::Bool=true)

    if verbose
        println("🚀 Starting enhanced constrained generation...")
        println("   Temperature: $(sampling_params.temperature)")
        println("   Top-k: $(sampling_params.top_k)")
        println("   Top-p: $(sampling_params.top_p)")
        println("   Seed: $(sampling_params.seed)")
    end

    # Initialize RNG with seed
    if sampling_params.seed == -1
        rng = Random.default_rng()
        Random.seed!(rng, abs(hash(prompt * string(time_ns()))))
    else
        rng = MersenneTwister(sampling_params.seed)
    end

    # Setup Outlines
    if isempty(hf_token)
        outlines_vocab = outlines.create_vocabulary(tokenizer)
    else
        outlines_vocab = outlines.create_vocabulary_with_token(tokenizer, hf_token)
    end

    schema_json = JSON3.write(schema)
    regex = outlines.regex_from_schema(schema_json)
    index_handle = outlines.create_index(regex, outlines_vocab)

    try
        # Process prompt
        llama.kv_cache_clear(model_context)
        llama.set_embeddings(model_context, false)

        prompt_tokens = llama.tokenize(vocab, prompt; add_special=true, parse_special=false)
        batch, keep = llama.build_batch(prompt_tokens)
        success = llama.decode(model_context, batch)

        if !success
            error("Failed to decode prompt")
        end

        # Initialize generation state
        current_state = outlines.index_initial_state(index_handle)
        current_pos = length(prompt_tokens)
        generated_text = ""
        vocab_size = llama.vocab_n_tokens(vocab)

        # Track recent tokens for repetition penalty
        recent_tokens = Int32[]

        if verbose
            println("🎯 Starting enhanced generation loop...")
        end

        # Enhanced generation loop
        for step in 1:max_tokens
            if sampling_params.verbose_sampling || verbose
                println("\n🔒 Step $step:")
                println("   State: $current_state")
                println("   Generated: '$generated_text'")
            end

            # Get allowed tokens from Outlines
            allowed_tokens = outlines.index_allowed_tokens(index_handle, current_state)
            if length(allowed_tokens) == 0
                if verbose
                    println("❌ No allowed tokens")
                end
                break
            end

            # Get base logits
            logits = llama.get_logits(model_context, vocab_size)

            # Apply Outlines constraints first
            masked_logits = fill(-Inf32, vocab_size)
            for token_id in allowed_tokens
                if token_id < vocab_size
                    masked_logits[token_id+1] = logits[token_id+1]
                end
            end

            # Apply sampling parameters
            if sampling_params.repeat_penalty != 1.0f0 && !isempty(recent_tokens)
                apply_repetition_penalty!(masked_logits, recent_tokens,
                    sampling_params.repeat_penalty, Int32(vocab_size))
            end

            apply_temperature!(masked_logits, sampling_params.temperature)

            if sampling_params.top_k > 0
                masked_logits = apply_top_k!(masked_logits, sampling_params.top_k)
            end

            if sampling_params.top_p < 1.0f0
                masked_logits = apply_top_p!(masked_logits, sampling_params.top_p)
            end

            if sampling_params.min_p > 0.0f0
                apply_min_p!(masked_logits, sampling_params.min_p)
            end

            # Sample token
            next_token = sample_token_enhanced(masked_logits, rng)

            # Update recent tokens for repetition penalty
            push!(recent_tokens, Int32(next_token))
            if length(recent_tokens) > sampling_params.repeat_last_n
                popfirst!(recent_tokens)
            end

            # Get token text and update generated string
            token_text = llama.vocab_get_text(vocab, Cint(next_token))
            generated_text *= token_text

            if sampling_params.verbose_sampling || verbose
                println("   🎲 Sampled: token $next_token -> '$token_text'")
            end

            # Update Outlines state
            next_state = outlines.index_next_state(index_handle, current_state, next_token)
            current_state = next_state

            # Check if final state
            is_final = outlines.index_is_final_state(index_handle, current_state)
            if is_final
                if verbose
                    println("🎉 Final state reached!")
                end
                break
            end

            # Update model context
            batch, keep = llama.build_single_token_batch(Cint(next_token), Cint(current_pos))
            success = llama.decode(model_context, batch)
            if !success
                if verbose
                    println("❌ Failed to decode token")
                end
                break
            end

            current_pos += 1
        end

        if verbose
            println("\n✅ Enhanced generation complete!")
            println("🏁 Raw result: '$generated_text'")
        end

        return generated_text

    finally
        outlines.free_index(index_handle)
        outlines.free_vocabulary(outlines_vocab)
    end
end

"""
    enhanced_mtmd_constrained_generation(prompt::String, img_paths::Vector{String}, schema::Dict, tokenizer::String; max_tokens::Int=50, hf_token::String="", model, model_context, vocab, mtmd_context, sampling_params::SamplingParams=balanced_params(), verbose::Bool=true)

Enhanced multimodal constrained generation with sampling parameters.

# Arguments
- `prompt::String`: Input prompt for generation
- `img_paths::Vector{String}`: Paths to image files
- `schema::Dict`: JSON schema defining the output structure
- `tokenizer::String`: HuggingFace model name for tokenizer

# Keyword Arguments
- `max_tokens::Int=50`: Maximum number of tokens to generate
- `hf_token::String=""`: HuggingFace authentication token (if needed)
- `model`: Pre-loaded model
- `model_context`: Pre-loaded model context
- `vocab`: Pre-loaded vocab
- `mtmd_context`: Pre-loaded MTMD context
- `sampling_params::SamplingParams=balanced_params()`: Sampling configuration for variety
- `verbose::Bool=true`: Print generation progress

# Returns
- `String`: Generated text

# Example
```julia
result = enhanced_mtmd_constrained_generation(
    "Describe this image in JSON format:",
    ["path/to/image.jpg"],
    schema,
    "google/gemma-3-4b-it",
    model=model,
    model_context=model_context,
    vocab=vocab,
    mtmd_context=mtmd_context,
    sampling_params=creative_params()
)
```
"""
function enhanced_mtmd_constrained_generation(prompt::String, img_paths::Vector{String}, schema::Dict, tokenizer::String;
    max_tokens::Int=50,
    hf_token::String="",
    model,
    model_context,
    vocab,
    mtmd_context,
    sampling_params::SamplingParams=balanced_params(),
    verbose::Bool=true)

    if verbose
        println("🚀 Starting enhanced multimodal constrained generation...")
        println("   Tokenizer: $tokenizer")
        println("   Images: $(length(img_paths))")
        println("   Max tokens: $max_tokens")
        println("   Temperature: $(sampling_params.temperature)")
        println("   Top-k: $(sampling_params.top_k)")
        println("   Top-p: $(sampling_params.top_p)")
        println("   Seed: $(sampling_params.seed)")
    end

    # Initialize RNG with seed
    if sampling_params.seed == -1
        rng = Random.default_rng()
        Random.seed!(rng, abs(hash(prompt * string(time_ns()) * join(img_paths))))
    else
        rng = MersenneTwister(sampling_params.seed)
    end

    # Step 1: Setup Outlines constraints
    if verbose
        println("🔧 Setting up Outlines constraints...")
    end

    # Create vocabulary handle
    if isempty(hf_token)
        outlines_vocab = outlines.create_vocabulary(tokenizer)
    else
        outlines_vocab = outlines.create_vocabulary_with_token(tokenizer, hf_token)
    end

    # Generate regex from schema
    schema_json = JSON3.write(schema)
    regex = outlines.regex_from_schema(schema_json)

    # Create index
    index_handle = outlines.create_index(regex, outlines_vocab)

    if verbose
        println("✅ Outlines setup complete")
    end

    try
        # Step 2: Multimodal encoding
        if verbose
            println("🔧 Processing multimodal input...")
        end

        # Clear KV cache and disable embeddings
        llama.kv_cache_clear(model_context)
        llama.set_embeddings(model_context, false)

        # Load image bitmaps
        image_bitmaps = [load_an_image_bitmap(path) for path in img_paths]
        if verbose
            println("📷 Loaded $(length(image_bitmaps)) images")
        end

        # Initialize chunks
        chunks = mtmd.input_chunks_init()

        # Create input text
        prompt_text = mtmd.create_input_text(prompt)

        # Tokenize multimodal input
        ret = mtmd.tokenize(mtmd_context, chunks, prompt_text, image_bitmaps)
        if ret != 0
            error("Multimodal tokenization failed with code $ret")
        end

        if verbose
            println("✅ Multimodal tokenization complete")
        end

        # Evaluate chunks to get initial context
        n_past = 0
        seq_id = 0
        n_batch = llama.n_batch(model_context)

        success, new_n_past = mtmd.helper_eval_chunks(mtmd_context, model_context, chunks,
            n_past, seq_id, n_batch, true)

        if !success
            error("Failed to evaluate multimodal chunks")
        end

        if verbose
            println("✅ Multimodal context processed")
            println("   Context length: $new_n_past tokens")
        end

        # Step 3: Initialize enhanced generation state
        current_state = outlines.index_initial_state(index_handle)
        current_pos = new_n_past  # Start from end of multimodal context
        generated_text = ""
        vocab_size = llama.vocab_n_tokens(vocab)

        # Track recent tokens for repetition penalty
        recent_tokens = Int32[]

        if verbose
            println("🎯 Starting enhanced constrained generation loop...")
            println("   Initial state: $current_state")
            println("   Starting position: $current_pos")
            println("   Vocab size: $vocab_size")
        end

        # Step 4: Enhanced constrained generation loop
        for step in 1:max_tokens
            if sampling_params.verbose_sampling || verbose
                println("\n🔒 Step $step:")
                println("   State: $current_state")
                println("   Generated: '$generated_text'")
                println("   Recent tokens: $(length(recent_tokens))")
            end

            # Get allowed tokens from Outlines
            allowed_tokens = outlines.index_allowed_tokens(index_handle, current_state)

            if length(allowed_tokens) == 0
                if verbose
                    println("❌ No allowed tokens")
                end
                break
            end

            if sampling_params.verbose_sampling || verbose
                println("   Allowed: $(length(allowed_tokens)) tokens")
            end

            # Get base logits from model
            logits = llama.get_logits(model_context, vocab_size)

            # Apply Outlines constraints first (this is critical)
            masked_logits = fill(-Inf32, vocab_size)
            allowed_count = 0
            for token_id in allowed_tokens
                if token_id < vocab_size
                    masked_logits[token_id+1] = logits[token_id+1]  # Julia 1-indexed
                    allowed_count += 1
                end
            end

            if sampling_params.verbose_sampling
                println("   Constrained to: $allowed_count valid tokens")
            end

            # Check for valid tokens after constraints
            valid_indices = findall(x -> x != -Inf32, masked_logits)
            if isempty(valid_indices)
                if verbose
                    println("❌ No valid tokens after Outlines constraints")
                end
                break
            end

            # Apply enhanced sampling parameters

            # 1. Repetition penalty (reduce likelihood of recent tokens)
            if sampling_params.repeat_penalty != 1.0f0 && !isempty(recent_tokens)
                apply_repetition_penalty!(masked_logits, recent_tokens,
                    sampling_params.repeat_penalty, Int32(vocab_size))
                if sampling_params.verbose_sampling
                    println("   Applied repetition penalty: $(sampling_params.repeat_penalty)")
                end
            end

            # 2. Temperature scaling (control randomness)
            apply_temperature!(masked_logits, sampling_params.temperature)
            if sampling_params.verbose_sampling
                println("   Applied temperature: $(sampling_params.temperature)")
            end

            # 3. Top-k filtering (only keep top k most likely tokens)
            if sampling_params.top_k > 0
                masked_logits = apply_top_k!(masked_logits, sampling_params.top_k)
                if sampling_params.verbose_sampling
                    remaining = count(x -> isfinite(x), masked_logits)
                    println("   Applied top-k ($(sampling_params.top_k)): $remaining tokens remaining")
                end
            end

            # 4. Top-p filtering (nucleus sampling - keep tokens until cumulative prob >= p)
            if sampling_params.top_p < 1.0f0
                masked_logits = apply_top_p!(masked_logits, sampling_params.top_p)
                if sampling_params.verbose_sampling
                    remaining = count(x -> isfinite(x), masked_logits)
                    println("   Applied top-p ($(sampling_params.top_p)): $remaining tokens remaining")
                end
            end

            # 5. Min-p filtering (remove tokens with prob < min_p * max_prob)
            if sampling_params.min_p > 0.0f0
                apply_min_p!(masked_logits, sampling_params.min_p)
                if sampling_params.verbose_sampling
                    remaining = count(x -> isfinite(x), masked_logits)
                    println("   Applied min-p ($(sampling_params.min_p)): $remaining tokens remaining")
                end
            end

            # Final check for valid tokens after all sampling filters
            final_valid = findall(x -> isfinite(x), masked_logits)
            if isempty(final_valid)
                if verbose
                    println("❌ No valid tokens after sampling filters")
                end
                break
            end

            # Sample token using enhanced method
            next_token = sample_token_enhanced(masked_logits, rng)

            # Update recent tokens tracking for repetition penalty
            push!(recent_tokens, Int32(next_token))
            if length(recent_tokens) > sampling_params.repeat_last_n
                popfirst!(recent_tokens)
            end

            # Get token text and update generated string
            token_text = llama.vocab_get_text(vocab, Cint(next_token))
            generated_text *= token_text

            if sampling_params.verbose_sampling || verbose
                println("   🎲 Sampled: token $next_token -> '$token_text'")
            end

            # Update Outlines state with the sampled token
            next_state = outlines.index_next_state(index_handle, current_state, next_token)
            current_state = next_state

            # Check if we've reached a final state in the schema
            is_final = outlines.index_is_final_state(index_handle, current_state)
            if is_final
                if verbose
                    println("🎉 Final state reached!")
                end
                break
            end

            # Update model context with the new token
            batch, keep = llama.build_single_token_batch(Cint(next_token), Cint(current_pos))
            success = llama.decode(model_context, batch)
            if !success
                if verbose
                    println("❌ Failed to decode token")
                end
                break
            end

            current_pos += 1
        end

        if verbose
            println("\n✅ Enhanced multimodal generation complete!")
            println("🏁 Raw result: '$generated_text'")
            println("📊 Final stats:")
            println("   - Generated $(length(generated_text)) characters")
            println("   - Used $(length(recent_tokens)) recent tokens for repetition penalty")
            println("   - Final sampling state: $current_state")
        end

        # Cleanup multimodal resources
        mtmd.input_chunks_free(chunks)
        for bitmap in image_bitmaps
            mtmd.bitmap_free(bitmap)
        end

        return generated_text

    finally
        # Always cleanup Outlines resources
        outlines.free_index(index_handle)
        outlines.free_vocabulary(outlines_vocab)
        if verbose
            println("🧹 Cleaned up Outlines resources")
        end
    end
end