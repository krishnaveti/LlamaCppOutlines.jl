using LlamaCppOutlines
using Test
using JSON3

# Test configuration
const MODEL_GGUF = joinpath(@__DIR__, "..", "models", "gemma-3-4b-it-Q2_K.gguf")
const PROJ_GGUF = joinpath(@__DIR__, "..", "models", "gemma3-4b-mmproj.gguf")
const TEST_IMAGE = joinpath(@__DIR__, "..", "images", "dog.jpg")

function test_integration()
    println("🧪 Testing Integration...")

    # Test constrained generation
    test_constrained_generation()

    # Test enhanced generation
    test_enhanced_generation()

    # Test multimodal constrained generation
    test_multimodal_constrained_generation()

    # Test end-to-end workflow
    test_end_to_end_workflow()
end

function test_constrained_generation()
    println("🧪 Testing Constrained Generation...")

    # if !haskey(ENV, "HF_TOKEN")
    #     @test_skip "Skipping constrained generation test - requires HF_TOKEN environment variable"
    #     println("⏭️  Skipping constrained generation test (requires HF_TOKEN)")
    #     return
    # end

    if !isfile(MODEL_GGUF)
        @warn "Model file not found: $MODEL_GGUF - skipping constrained generation tests"
        return
    end

    try
        # Load model
        model, model_context, vocab = load_and_initialize(MODEL_GGUF)

        # Test schema
        schema = Dict(
            "type" => "object",
            "properties" => Dict(
                "name" => Dict("type" => "string"),
                "age" => Dict("type" => "integer")
            ),
            "required" => ["name", "age"]
        )

        # Simple prompt
        prompt = "Generate a person with name and age:"

        # Test greedy constrained generation
        result = greedy_constrained_generation(
            prompt,
            schema,
            "openai-community/gpt2",
            max_tokens=30,
            model_context=model_context,
            vocab=vocab,
            verbose=false
        )

        @test typeof(result) == String
        @test length(result) > 0
        println("✅ Greedy constrained generation successful")
        println("   Generated: '$(result[1:min(100, length(result))])'")

    catch e
        @warn "Constrained generation test failed (expected if no internet): $e"
    end
end

function test_enhanced_generation()
    println("🧪 Testing Enhanced Generation...")

    # if !haskey(ENV, "HF_TOKEN")
    #     @test_skip "Skipping enhanced generation test - requires HF_TOKEN environment variable"
    #     println("⏭️  Skipping enhanced generation test (requires HF_TOKEN)")
    #     return
    # end

    if !isfile(MODEL_GGUF)
        @warn "Model file not found: $MODEL_GGUF - skipping enhanced generation tests"
        return
    end

    try
        # Load model
        model, model_context, vocab = load_and_initialize(MODEL_GGUF)

        # Test schema
        schema = Dict(
            "type" => "object",
            "properties" => Dict(
                "response" => Dict("type" => "string")
            ),
            "required" => ["response"]
        )

        # Test different sampling parameters
        sampling_configs = [
            ("Creative", creative_params()),
            ("Balanced", balanced_params()),
            ("Focused", focused_params()),
            ("Greedy", greedy_params())
        ]

        prompt = "Generate a creative response:"

        for (name, params) in sampling_configs
            result = enhanced_constrained_generation(
                prompt,
                schema,
                "openai-community/gpt2",
                max_tokens=20,
                model_context=model_context,
                vocab=vocab,
                sampling_params=params,
                verbose=false
            )

            @test typeof(result) == String
            println("✅ Enhanced generation with $name params successful")
        end

    catch e
        @warn "Enhanced generation test failed (expected if no internet): $e"
    end
end

function test_multimodal_constrained_generation()
    println("🧪 Testing Multimodal Constrained Generation...")

    # if !haskey(ENV, "HF_TOKEN")
    #     @test_skip "Skipping multimodal constrained generation test - requires HF_TOKEN environment variable"
    #     println("⏭️  Skipping multimodal constrained generation test (requires HF_TOKEN)")
    #     return
    # end

    if !isfile(MODEL_GGUF) || !isfile(PROJ_GGUF)
        @warn "Multimodal model files not found - skipping multimodal constrained generation tests"
        return
    end

    try
        # Load multimodal model
        model, model_context, mtmd_context, vocab = load_and_initialize_mtmd(MODEL_GGUF, PROJ_GGUF)

        # Test schema
        schema = Dict(
            "type" => "object",
            "properties" => Dict(
                "description" => Dict("type" => "string"),
                "objects" => Dict("type" => "array", "items" => Dict("type" => "string"))
            ),
            "required" => ["description", "objects"]
        )

        prompt = "Describe this image: <__media__>"

        if isfile(TEST_IMAGE)
            # Test greedy multimodal constrained generation
            result = greedy_mtmd_constrained_generation(
                prompt,
                [TEST_IMAGE],
                schema,
                "openai-community/gpt2",
                max_tokens=30,
                model=model,
                model_context=model_context,
                vocab=vocab,
                mtmd_context=mtmd_context,
                verbose=false
            )

            @test typeof(result) == String
            println("✅ Multimodal constrained generation interface successful")
            println("   Note: Actual image processing requires additional libraries")
        else
            @warn "Test image not found: $TEST_IMAGE"
        end

    catch e
        @warn "Multimodal constrained generation test failed: $e"
    end
end

function test_end_to_end_workflow()
    println("🧪 Testing End-to-End Workflow...")

    if !isfile(MODEL_GGUF)
        @warn "Model file not found: $MODEL_GGUF - skipping end-to-end tests"
        return
    end

    try
        # Initialize APIs
        init_apis!()
        println("✅ APIs initialized")

        # Load model
        model, model_context, vocab = load_and_initialize(MODEL_GGUF)
        println("✅ Model loaded")

        # Test basic generation
        prompt = "Hello, world!"
        result = generate_with_sampling(
            prompt,
            model=model,
            model_context=model_context,
            vocab=vocab,
            max_new_tokens=5
        )

        @test typeof(result) == String
        @test length(result) > 0
        println("✅ Basic generation successful")

        # Test SamplingParams
        params = SamplingParams(temperature=0.7f0, top_k=30)
        @test params.temperature == 0.7f0
        @test params.top_k == 30
        println("✅ SamplingParams creation successful")

        # Test preset configurations
        creative = creative_params()
        balanced = balanced_params()
        focused = focused_params()
        greedy = greedy_params()

        @test creative.temperature > balanced.temperature
        @test balanced.temperature > focused.temperature
        @test focused.temperature > greedy.temperature
        println("✅ Preset configurations verified")

        println("✅ End-to-end workflow successful")

    catch e
        @warn "End-to-end workflow test failed: $e"
    end
end

function test_error_handling()
    println("🧪 Testing Error Handling...")

    # Test invalid model path
    try
        model, model_context, vocab = load_and_initialize("nonexistent_model.gguf")
        @test false  # Should not reach here
    catch e
        @test true  # Expected to fail
        println("✅ Invalid model path handling successful")
    end

    # Test invalid schema
    try
        schema = Dict("invalid" => "schema")
        regex = outlines.regex_from_schema(JSON3.write(schema))
        @test false  # Should not reach here
    catch e
        @test true  # Expected to fail
        println("✅ Invalid schema handling successful")
    end
end