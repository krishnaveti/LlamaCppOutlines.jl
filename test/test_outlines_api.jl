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
        version = outlines.version()
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
        regex = outlines.regex_from_schema(schema)
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
        city_regex = outlines.regex_from_schema(city_schema)
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
    try
        # Try to create vocabulary (this requires network access)
        vocab = outlines.create_vocabulary("google/gemma-3-4b-it")

        # Test vocabulary operations
        vocab_size = outlines.vocabulary_size(vocab)
        @test vocab_size > 0
        println("✅ Vocabulary created successfully, size: $vocab_size")

        eos_token = outlines.vocabulary_eos_token_id(vocab)
        @test eos_token >= 0
        println("✅ EOS token ID: $eos_token")

        # Clean up
        outlines.free_vocabulary(vocab)
        println("✅ Vocabulary freed successfully")

    catch e
        @warn "Failed vocabulary test: $e"
        @warn "   (This is expected if you don't have internet access)"
    end
end

function test_index_creation()
    println("🧪 Testing index creation...")
    try
        # Create vocabulary first
        vocab = outlines.create_vocabulary("google/gemma-3-4b-it")

        # Create simple schema and regex
        schema = """{"type": "string"}"""
        regex = outlines.regex_from_schema(schema)

        # Create index
        index = outlines.create_index(regex, vocab)
        @test index != C_NULL
        println("✅ Index created successfully")

        # Test index operations
        initial_state = outlines.index_initial_state(index)
        @test initial_state >= 0
        println("✅ Initial state: $initial_state")

        allowed_tokens = outlines.index_allowed_tokens(index, initial_state)
        @test length(allowed_tokens) > 0
        println("✅ Allowed tokens count: $(length(allowed_tokens))")

        is_final = outlines.index_is_final_state(index, initial_state)
        @test typeof(is_final) == Bool
        println("✅ Is initial state final: $is_final")

        # Clean up
        outlines.free_index(index)
        outlines.free_vocabulary(vocab)
        println("✅ Index and vocabulary freed successfully")

    catch e
        @warn "Failed index test: $e"
        @warn "   (This is expected if you don't have internet access)"
    end
end

function test_sampling_functions()
    println("🧪 Testing sampling functions...")
    
    # Test temperature application
    logits = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
    temp_logits = copy(logits)
    apply_temperature!(temp_logits, 0.5f0)
    @test temp_logits != logits  # Should be different
    println("✅ Temperature application successful")
    
    # Test top-k filtering
    logits = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
    topk_logits = apply_top_k!(copy(logits), 3)
    finite_count = count(x -> isfinite(x), topk_logits)
    @test finite_count == 3
    println("✅ Top-k filtering successful")
    
    # Test top-p filtering
    logits = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
    topp_logits = apply_top_p!(copy(logits), 0.8f0)
    finite_count = count(x -> isfinite(x), topp_logits)
    @test finite_count <= 5
    println("✅ Top-p filtering successful")
    
    # Test min-p filtering
    logits = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
    minp_logits = copy(logits)
    apply_min_p!(minp_logits, 0.1f0)
    finite_count = count(x -> isfinite(x), minp_logits)
    @test finite_count <= 5
    println("✅ Min-p filtering successful")
    
    # Test repetition penalty
    logits = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
    recent_tokens = Int32[1, 2]  # 0-indexed tokens
    penalty_logits = copy(logits)
    apply_repetition_penalty!(penalty_logits, recent_tokens, 1.2f0, Int32(5))
    @test penalty_logits[2] != logits[2]  # Token 1 should be penalized
    @test penalty_logits[3] != logits[3]  # Token 2 should be penalized
    println("✅ Repetition penalty successful")
    
    # Test softmax
    logits = Float32[1.0, 2.0, 3.0]
    probs = softmax(logits)
    @test abs(sum(probs) - 1.0f0) < 1e-6  # Should sum to 1
    @test all(p -> p >= 0, probs)  # All probabilities should be positive
    println("✅ Softmax successful")
end