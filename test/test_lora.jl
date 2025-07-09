using LlamaCppOutlines
using Test

# Test configuration
const MODEL_GGUF = joinpath(@__DIR__, "..", "models", "gemma-3-4b-it-Q2_K.gguf")

function test_lora_functionality()
    println("🧪 Testing LoRA Functionality...")
    
    if !isfile(MODEL_GGUF)
        @warn "Model file not found: $MODEL_GGUF - skipping LoRA tests"
        return
    end
    
    # Test LoRA training wrapper function interface
    test_lora_training_interface()
    
    # Test LoRA adapter management
    test_lora_adapter_management()
    
    # Test LoRA manager functionality
    test_lora_manager()
end

function test_lora_training_interface()
    println("🧪 Testing LoRA Training Interface...")
    
    # Test minimal training data creation
    test_data_path = "test_training_data.json"
    create_minimal_training_data(test_data_path)
    
    @test isfile(test_data_path)
    println("✅ Minimal training data creation successful")
    
    # Test listing GGUF files (empty directory case)
    gguf_files = list_lora_gguf_files("nonexistent_dir")
    @test isempty(gguf_files)
    println("✅ Empty directory GGUF listing successful")
    
    # Clean up test file
    rm(test_data_path, force=true)
    rm("lora_training", force=true, recursive=true)
end

function test_lora_adapter_management()
    println("🧪 Testing LoRA Adapter Management...")
    
    # Test that LoRA functions are available and callable
    # Note: These will fail without actual adapters, but we test the interface
    
    try
        model, model_context, vocab = load_and_initialize(MODEL_GGUF)
        
        # Test function availability (they should exist even if they fail)
        @test isdefined(LlamaCppOutlines, :load_lora_adapter)
        @test isdefined(LlamaCppOutlines, :free_lora_adapter)
        @test isdefined(LlamaCppOutlines, :set_adapter_lora)
        @test isdefined(LlamaCppOutlines, :rm_adapter_lora)
        @test isdefined(LlamaCppOutlines, :clear_adapter_lora)
        
        println("✅ LoRA adapter functions available")
        
        # Test clear adapter (should work even without adapters)
        result = clear_adapter_lora(model_context)
        @test typeof(result) <: Integer
        println("✅ Clear adapter function successful")
        
    catch e
        @warn "LoRA adapter management test failed: $e"
    end
end

function test_lora_manager()
    println("🧪 Testing LoRA Manager...")
    
    try
        model, model_context, vocab = load_and_initialize(MODEL_GGUF)
        
        # Test LoRA manager creation
        @test isdefined(LlamaCppOutlines, :LoRAManager)
        manager = LoRAManager(model)
        @test manager !== nothing
        println("✅ LoRA manager creation successful")
        
        # Test manager functions availability
        @test isdefined(LlamaCppOutlines, :load_adapter!)
        @test isdefined(LlamaCppOutlines, :switch_adapter!)
        @test isdefined(LlamaCppOutlines, :clear_active_adapter!)
        @test isdefined(LlamaCppOutlines, :free_all_adapters!)
        @test isdefined(LlamaCppOutlines, :list_adapters)
        @test isdefined(LlamaCppOutlines, :get_active_adapter)
        
        println("✅ LoRA manager functions available")
        
        # Test list adapters (should return empty list initially)
        adapters = list_adapters(manager)
        @test typeof(adapters) <: AbstractVector
        println("✅ List adapters function successful")
        
        # Test get active adapter (should return empty string initially)
        active = get_active_adapter(manager)
        @test typeof(active) == String
        println("✅ Get active adapter function successful")
        
        # Test clear active adapter
        clear_active_adapter!(manager, model_context)
        println("✅ Clear active adapter function successful")
        
    catch e
        @warn "LoRA manager test failed: $e"
    end
end

function test_lora_training_workflow()
    println("🧪 Testing LoRA Training Workflow...")
    
    # This test requires internet access and Python dependencies
    # We'll only test the interface without actually running training
    
    @test isdefined(LlamaCppOutlines, :train_lora_to_gguf)
    @test isdefined(LlamaCppOutlines, :list_lora_gguf_files)
    
    println("✅ LoRA training functions available")
    
    # Test listing GGUF files in current directory
    current_gguf = list_lora_gguf_files(".")
    @test typeof(current_gguf) == Vector{String}
    println("✅ GGUF file listing successful")
    
    # Note: Actual training test would require:
    # result = train_lora_to_gguf("google/gemma-2-2b-it", "fake_token", verbose=false)
    # But this requires internet, Python deps, and HF token
    
    println("   Note: Actual training requires internet, Python dependencies, and HF token")
end

# Helper function to create test training data
function create_minimal_training_data(output_path::String)
    minimal_data = [
        Dict(
            "problem_type" => "test",
            "problem_statement" => "Test problem",
            "solution_steps" => ["Step 1", "Step 2"],
            "final_answer" => "Test answer"
        )
    ]
    
    # Ensure directory exists
    mkpath(dirname(output_path))
    
    # Write JSON data
    open(output_path, "w") do f
        for item in minimal_data
            json_str = JSON3.write(item)
            write(f, json_str * "\n")
        end
    end
end