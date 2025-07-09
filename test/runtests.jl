using Test
using LlamaCppOutlines

# Initialize APIs
LlamaCppOutlines.init_apis!()

# Include all test files
include("test_basic_api.jl")
include("test_multimodal_api.jl")
include("test_outlines_api.jl")
include("test_lora.jl")
include("test_integration.jl")

# Run all tests
@testset "LlamaCppOutlines.jl Tests" begin
    @testset "Basic API Tests" begin
        test_basic_api()
    end
    
    @testset "Multimodal API Tests" begin
        test_multimodal_api()
    end
    
    @testset "Outlines API Tests" begin
        test_outlines_api()
    end
    
    @testset "LoRA Tests" begin
        test_lora_functionality()
    end
    
    @testset "Integration Tests" begin
        test_integration()
    end
end