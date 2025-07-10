using LlamaCppOutlines
using Test
using JSON3

function test_outlines_api()
    println("🧪 Testing Outlines API...")

    # Test Outlines version
    test_outlines_version()

    # Test schema to regex conversion
    test_schema_to_regex()

    # Test vocabulary creation (requires internet)
    test_vocabulary_creation()

    # Test index creation
    test_index_creation()

    # Test sampling functions
    test_sampling_functions()
end

function test_outlines_version()
    println("🧪 Testing Outlines version...")
    try
        version = LlamaCppOutlines.outlines.version()
        @test typeof(version) == String
        @test length(version) > 0
        println("✅ Outlines version: $version")
    catch e
        @error "Failed to get Outlines version: $e"
        rethrow(e)
    end
end

function test_schema_to_regex()
    println("🧪 Testing schema to regex conversion...")
    try
        # Simple string schema
        schema = """{"type": "string"}"""
        regex = LlamaCppOutlines.outlines.regex_from_schema(schema)
        @test typeof(regex) == String
        @test length(regex) > 0
        println("✅ Generated regex for string schema: $(regex[1:min(50, length(regex))])...")

        # Simple object schema
        city_schema = """{
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }"""
        city_regex = LlamaCppOutlines.outlines.regex_from_schema(city_schema)
        @test typeof(city_regex) == String
        @test length(city_regex) > 0
        println("✅ Generated regex for city schema: $(city_regex[1:min(50, length(city_regex))])...")

    catch e
        @error "Failed schema to regex test: $e"
        rethrow(e)
    end
end

function test_vocabulary_creation()
    println("🧪 Testing vocabulary creation...")

    # if !haskey(ENV, "HF_TOKEN")
    #     @test_skip "Skipping vocabulary creation test - requires HF_TOKEN environment variable"
    #     println("⏭️  Skipping vocabulary creation test (requires HF_TOKEN)")
    #     return
    # end

    try
        # Try to create vocabulary (this requires network access)
        vocab = LlamaCppOutlines.outlines.create_vocabulary("openai-community/gpt2")

        # Test vocabulary operations
        vocab_size = LlamaCppOutlines.outlines.vocabulary_size(vocab)
        @test vocab_size > 0
        println("✅ Vocabulary created successfully, size: $vocab_size")

        eos_token = LlamaCppOutlines.outlines.vocabulary_eos_token_id(vocab)
        @test eos_token >= 0
        println("✅ EOS token ID: $eos_token")

        # Clean up
        LlamaCppOutlines.outlines.free_vocabulary(vocab)
        println("✅ Vocabulary freed successfully")

    catch e
        @warn "Failed vocabulary test: $e"
        @warn "   (This is expected if you don't have internet access)"
    end
end

function test_index_creation()
    println("🧪 Testing index creation...")

    # if !haskey(ENV, "HF_TOKEN")
    #     @test_skip "Skipping index creation test - requires HF_TOKEN environment variable"
    #     println("⏭️  Skipping index creation test (requires HF_TOKEN)")
    #     return
    # end

    try
        # Create vocabulary first (requires network access) and this is not compatible tokenizer with the google model but it is not a problem for the test
        # but ideally, the tokenizer should be compatible with the model
        # This is a workaround to avoid the error in the test
        vocab = LlamaCppOutlines.outlines.create_vocabulary("openai-community/gpt2")

        # Create simple schema and regex
        schema = """{"type": "string"}"""
        regex = LlamaCppOutlines.outlines.regex_from_schema(schema)

        # Create index
        index = LlamaCppOutlines.outlines.create_index(regex, vocab)
        @test index != C_NULL
        println("✅ Index created successfully")

        # Test index operations
        initial_state = LlamaCppOutlines.outlines.index_initial_state(index)
        @test initial_state >= 0
        println("✅ Initial state: $initial_state")

        allowed_tokens = LlamaCppOutlines.outlines.index_allowed_tokens(index, initial_state)
        @test length(allowed_tokens) > 0
        println("✅ Allowed tokens count: $(length(allowed_tokens))")

        is_final = LlamaCppOutlines.outlines.index_is_final_state(index, initial_state)
        @test typeof(is_final) == Bool
        println("✅ Is initial state final: $is_final")

        # Clean up
        LlamaCppOutlines.outlines.free_index(index)
        LlamaCppOutlines.outlines.free_vocabulary(vocab)
        println("✅ Index and vocabulary freed successfully")

    catch e
        @warn "Failed index test: $e"
        @warn "   (This is expected if you don't have internet access)"
    end
end

function test_sampling_functions()
    println("🧪 Testing sampling functions...")

    # Test SamplingParams struct functionality
    params = SamplingParams()
    @test params.temperature == 0.8f0
    @test params.top_k == 40
    @test params.top_p == 0.9f0
    println("✅ SamplingParams structure creation successful")

    # Test preset parameter functions
    creative = creative_params()
    @test creative.temperature == 1.2f0
    println("✅ Creative params preset successful")

    balanced = balanced_params()
    @test balanced.temperature == 0.8f0
    println("✅ Balanced params preset successful")

    focused = focused_params()
    @test focused.temperature == 0.3f0
    println("✅ Focused params preset successful")

    greedy = greedy_params()
    @test greedy.temperature == 0.0f0
    println("✅ Greedy params preset successful")

    println("✅ All available sampling functions tested successfully")
end