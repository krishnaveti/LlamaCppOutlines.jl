using LlamaCppOutlines
using Test

# Test configuration
const MODEL_GGUF = joinpath(@__DIR__, "..", "models", "gemma-3-4b-it-Q2_K.gguf")
const LLAMA_DEFAULT_SEED = 0xDEADBEEF

function test_basic_api()
    println("🧪 Testing Basic LLaMA API...")

    # Skip if model doesn't exist
    if !isfile(MODEL_GGUF)
        @warn "Model file not found: $MODEL_GGUF - skipping basic API tests"
        return
    end

    try
        # Test model loading
        model, model_context, vocab = load_and_initialize(MODEL_GGUF)
        @test model != C_NULL
        @test model_context != C_NULL
        @test vocab != C_NULL
        println("✅ Model loading successful")

        # Test basic generation
        prompt = "What is the capital of France?"
        result = generate_with_sampling(prompt,
            model=model,
            model_context=model_context,
            vocab=vocab,
            max_new_tokens=10)

        @test typeof(result) == String
        @test length(result) > 0
        println("✅ Basic generation successful")
        println("   Generated: '$(result[1:min(50, length(result))])'")

        # Test sampler chain initialization
        chain = init_sampler_chain()
        @test chain != C_NULL
        println("✅ Sampler chain initialization successful")

        # Clean up
        # llama.sampler_free(chain)

    catch e
        @error "Basic API test failed: $e"
        rethrow(e)
    end
end

function test_sampling_params()
    println("🧪 Testing Sampling Parameters...")

    # Test SamplingParams struct creation
    params = SamplingParams()
    @test params.temperature == 0.8f0
    @test params.top_k == 40
    @test params.top_p == 0.9f0
    println("✅ Default SamplingParams created")

    # Test preset configurations
    creative = creative_params()
    @test creative.temperature == 1.2f0
    @test creative.top_k == 100
    println("✅ Creative params preset")

    balanced = balanced_params()
    @test balanced.temperature == 0.8f0
    @test balanced.top_k == 40
    println("✅ Balanced params preset")

    focused = focused_params()
    @test focused.temperature == 0.3f0
    @test focused.top_k == 20
    println("✅ Focused params preset")

    greedy = greedy_params()
    @test greedy.temperature == 0.0f0
    @test greedy.top_k == 1
    println("✅ Greedy params preset")
end

function test_helper_functions()
    println("🧪 Testing Helper Functions...")

    # Test BPE token cleaning
    test_text = "Hello▁world▁this▁is▁a▁test"
    cleaned = clean_bpe_tokens(test_text)
    @test cleaned == "Hello world this is a test"
    println("✅ BPE token cleaning successful")

    # Test context clearing (requires loaded model)
    if isfile(MODEL_GGUF)
        model, model_context, vocab = load_and_initialize(MODEL_GGUF)
        clear_context_state(model_context)
        println("✅ Context clearing successful")
    end
end

# Clean BPE tokens function for testing
function clean_bpe_tokens(text::String)
    # Replace common BPE tokens with spaces
    cleaned = text
    cleaned = replace(cleaned, "▁" => " ")      # SentencePiece space token
    cleaned = replace(cleaned, "Ġ" => " ")      # GPT-2 style space token  
    cleaned = replace(cleaned, "##" => "")      # BERT subword tokens
    cleaned = replace(cleaned, "</w>" => "")    # Some tokenizers use this
    cleaned = replace(cleaned, "<unk>" => "")   # Unknown tokens
    cleaned = replace(cleaned, "�" => "")       # Invalid UTF-8 replacement char

    return cleaned
end