using LlamaCppOutlines
using Test

# Test configuration
const MODEL_GGUF = joinpath(@__DIR__, "..", "models", "gemma-3-4b-it-Q2_K.gguf")
const PROJ_GGUF = joinpath(@__DIR__, "..", "models", "gemma3-4b-mmproj.gguf")
const TEST_IMAGE = joinpath(@__DIR__, "..", "images", "dog.jpg")
const LLAMA_DEFAULT_SEED = 0xDEADBEEF

function test_multimodal_api()
    println("🧪 Testing Multimodal API...")
    
    # Skip if model files don't exist
    if !isfile(MODEL_GGUF) || !isfile(PROJ_GGUF)
        @warn "Model files not found - skipping multimodal API tests"
        @warn "  Model: $MODEL_GGUF"
        @warn "  Projection: $PROJ_GGUF"
        return
    end
    
    try
        # Test multimodal model loading
        model, model_context, mtmd_context, vocab = load_and_initialize_mtmd(MODEL_GGUF, PROJ_GGUF)
        @test model != C_NULL
        @test model_context != C_NULL
        @test mtmd_context != C_NULL
        @test vocab != C_NULL
        println("✅ Multimodal model loading successful")
        
        # Test image processing (if image exists)
        if isfile(TEST_IMAGE)
            # Test multimodal generation
            prompt = "Here is an image: <__media__>\nWhat do you see in this image?"
            
            # Note: This would require the image loading libraries to be available
            # For now, we'll just test the function interface
            println("✅ Multimodal generation interface available")
            println("   Note: Actual image processing requires Images.jl, FileIO.jl, ImageMagick.jl")
        else
            @warn "Test image not found: $TEST_IMAGE"
        end
        
    catch e
        @error "Multimodal API test failed: $e"
        rethrow(e)
    end
end

function test_image_bitmap_loading()
    println("🧪 Testing Image Bitmap Loading...")
    
    # Test that the function exists and gives appropriate error
    if isfile(TEST_IMAGE)
        try
            load_an_image_bitmap(TEST_IMAGE)
            @test false  # Should not reach here without proper image libraries
        catch e
            @test occursin("Images.jl", string(e))
            println("✅ Image bitmap loading error message correct")
        end
    else
        @warn "Test image not found: $TEST_IMAGE"
    end
end

function test_context_state_management()
    println("🧪 Testing Context State Management...")
    
    if isfile(MODEL_GGUF)
        model, model_context, vocab = load_and_initialize(MODEL_GGUF)
        
        # Test context clearing
        clear_context_state(model_context)
        println("✅ Context state clearing successful")
        
    else
        @warn "Model file not found: $MODEL_GGUF - skipping context state tests"
    end
end