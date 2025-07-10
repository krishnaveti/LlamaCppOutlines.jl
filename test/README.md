# LlamaCppOutlines Test Suite

This directory contains the test suite for the LlamaCppOutlines.jl package.

## Running Tests

To run all tests:

```bash
julia --project=. test/runtests.jl
```

Or using the Julia package manager:

```julia
julia> using Pkg
julia> Pkg.test()
```

## HF_TOKEN Requirement

Some tests in this suite require access to Hugging Face models that are token-gated. These tests will be **automatically skipped** if the `HF_TOKEN` environment variable is not set.

### Tests that require HF_TOKEN:

- **Outlines API Tests** (`test_outlines_api.jl`):
  - `test_vocabulary_creation()` - Creates vocabulary from `google/gemma-3-4b-it`
  - `test_index_creation()` - Creates index using the vocabulary

- **Integration Tests** (`test_integration.jl`):
  - `test_constrained_generation()` - Tests constrained generation with Gemma model
  - `test_enhanced_generation()` - Tests enhanced generation with various sampling parameters
  - `test_multimodal_constrained_generation()` - Tests multimodal constrained generation

- **LoRA Tests** (`test_lora.jl`):
  - `test_lora_training_workflow()` - Tests LoRA training interface (actual training commented out)

### Setting up HF_TOKEN

To run the full test suite including tests that require Hugging Face model access:

1. **Get a Hugging Face token**: Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

2. **Set the environment variable**:
   ```bash
   # Linux/macOS
   export HF_TOKEN="your_token_here"
   
   # Windows
   set HF_TOKEN=your_token_here
   ```

3. **Run tests**:
   ```bash
   julia --project=. test/runtests.jl
   ```

### What happens without HF_TOKEN

If `HF_TOKEN` is not set, affected tests will:
- Display a message: `⏭️  Skipping [test name] (requires HF_TOKEN)`
- Use `@test_skip` to mark the test as skipped rather than failed
- Continue with other tests that don't require the token

## Test Categories

- **Basic API Tests**: Core functionality, model loading, sampling parameters
- **Multimodal API Tests**: Multimodal model loading and image processing interfaces
- **Outlines API Tests**: Schema-to-regex conversion, vocabulary creation, index operations
- **LoRA Tests**: LoRA adapter management, training interface
- **Integration Tests**: End-to-end workflows combining multiple components

## Dependencies

Some tests may require additional dependencies:
- Model files in the `models/` directory
- Image files in the `images/` directory
- Python dependencies for LoRA training
- Internet access for downloading tokenizers and models

## Notes

- Model files are expected to be in the `../models/` directory relative to the test files
- Tests are designed to gracefully handle missing files and dependencies
- Network-dependent tests include appropriate error handling and warnings