# ─────────────────────────────────────────────────────────────────────────────
# LoRA Training Functions
# ─────────────────────────────────────────────────────────────────────────────

using JSON3

"""
    train_lora_to_gguf(model_name::String, hf_token::String; output_dir::String="lora_gguf", training_data::String="lora_training/econ_fewshots.json", verbose::Bool=true)

Train a LoRA adapter and convert it to GGUF format using the Python training script.

# Arguments
- `model_name::String`: HuggingFace model name (e.g., "google/gemma-2-2b-it")
- `hf_token::String`: HuggingFace authentication token

# Keyword Arguments
- `output_dir::String="lora_gguf"`: Directory to save the final GGUF file
- `training_data::String="lora_training/econ_fewshots.json"`: Path to training data JSON file
- `verbose::Bool=true`: Print training progress

# Returns
- `String`: Path to the generated GGUF file if successful, empty string if failed

# Example
```julia
# Train LoRA adapter and convert to GGUF
gguf_path = train_lora_to_gguf(
    "google/gemma-2-2b-it",
    "your_hf_token_here",
    output_dir="my_lora_models"
)

if !isempty(gguf_path)
    println("LoRA GGUF created at: \$gguf_path")
    
    # Load and use the adapter
    adapter = load_lora_adapter(model, gguf_path)
    # ... use adapter
end
```

# Notes
- Requires Python with transformers, peft, torch, datasets, huggingface_hub, python-dotenv
- Will automatically install missing Python packages
- Creates intermediate directories for checkpoints and adapters
- Final GGUF file is saved in the specified output directory
"""
function train_lora_to_gguf(model_name::String, hf_token::String; 
    output_dir::String="lora_gguf", 
    training_data::String="lora_training/econ_fewshots.json",
    verbose::Bool=true)
    
    if verbose
        println("🚀 Starting LoRA training and GGUF conversion...")
        println("   Model: $model_name")
        println("   Output directory: $output_dir")
        println("   Training data: $training_data")
    end
    
    # Ensure output directory exists
    mkpath(output_dir)
    
    # Get the path to the Python script
    script_path = joinpath(@__DIR__, "..", "lib", "lora_to_gguf.py")
    
    if !isfile(script_path)
        @error "LoRA training script not found: $script_path"
        return ""
    end
    
    if verbose
        println("🔧 Installing/updating Python dependencies...")
    end
    
    # Install required Python packages
    pip_packages = [
        "torch",
        "transformers", 
        "peft",
        "datasets",
        "huggingface_hub",
        "python-dotenv",
        "accelerate"
    ]
    
    try
        # Install packages
        for package in pip_packages
            pip_cmd = `python -m pip install $package --upgrade`
            if verbose
                println("   Installing $package...")
            end
            result = run(pipeline(pip_cmd, stdout=devnull, stderr=devnull))
        end
        
        if verbose
            println("✅ Python dependencies installed/updated")
        end
    catch e
        @warn "Failed to install some Python packages: $e"
        @warn "Continuing anyway - packages might already be installed"
    end
    
    if verbose
        println("🔧 Setting up environment...")
    end
    
    # Create .env file with the HF token
    env_file = joinpath(pwd(), ".env")
    open(env_file, "w") do f
        write(f, "outlines_core=$hf_token\n")
    end
    
    # Ensure training data directory exists
    training_data_dir = dirname(training_data)
    if !isempty(training_data_dir)
        mkpath(training_data_dir)
    end
    
    # Create minimal training data if it doesn't exist
    if !isfile(training_data)
        if verbose
            println("⚠️  Training data not found, creating minimal example...")
        end
        create_minimal_training_data(training_data)
    end
    
    if verbose
        println("🚀 Starting Python LoRA training script...")
    end
    
    try
        # Set environment variables
        env_vars = Dict(
            "PYTHONPATH" => get(ENV, "PYTHONPATH", ""),
            "HF_TOKEN" => hf_token,
            "MODEL_NAME" => model_name
        )
        
        # Run the Python script
        python_cmd = `python $script_path`
        
        if verbose
            println("   Command: $python_cmd")
            println("   Working directory: $(pwd())")
        end
        
        # Run with output capture
        result = run(pipeline(python_cmd, stdout=verbose ? stdout : devnull, stderr=verbose ? stderr : devnull), env=env_vars)
        
        if verbose
            println("✅ Python script completed successfully")
        end
        
        # Look for the generated GGUF file
        expected_gguf_names = [
            "economics_lora_gemma2_2b.gguf",
            "economics_lora.gguf"
        ]
        
        gguf_file = ""
        # First try specific expected names
        for name in expected_gguf_names
            if isfile(name)
                gguf_file = name
                break
            end
        end
        
        # If not found, look for any .gguf file in current directory
        if isempty(gguf_file)
            for file in readdir(pwd())
                if endswith(lowercase(file), ".gguf")
                    gguf_file = file
                    break
                end
            end
        end
        
        if isempty(gguf_file)
            @warn "No GGUF file found after training"
            return ""
        end
        
        # Move GGUF file to output directory
        final_path = joinpath(output_dir, basename(gguf_file))
        if gguf_file != final_path
            mv(gguf_file, final_path, force=true)
        end
        
        if verbose
            println("🎉 LoRA training and GGUF conversion completed!")
            println("   Final GGUF file: $final_path")
            file_size = round(stat(final_path).size / (1024*1024), digits=1)
            println("   File size: $(file_size) MB")
        end
        
        # Clean up temporary files
        try
            rm(env_file, force=true)
        catch
            # Ignore cleanup errors
        end
        
        return final_path
        
    catch e
        @error "LoRA training failed: $e"
        
        # Clean up on failure
        try
            rm(env_file, force=true)
        catch
            # Ignore cleanup errors
        end
        
        return ""
    end
end

"""
Create minimal training data for LoRA training if none exists.
"""
function create_minimal_training_data(output_path::String)
    minimal_data = [
        Dict(
            "problem_type" => "microeconomics",
            "problem_statement" => "Calculate the marginal utility of consuming one more apple if the total utility increases from 10 to 15 utils.",
            "solution_steps" => [
                "Marginal utility = Change in total utility / Change in quantity",
                "MU = (15 - 10) / (1 - 0) = 5 utils"
            ],
            "final_answer" => "The marginal utility is 5 utils."
        ),
        Dict(
            "problem_type" => "macroeconomics", 
            "problem_statement" => "If GDP increases from \$1000 billion to \$1100 billion, what is the growth rate?",
            "solution_steps" => [
                "Growth rate = (New GDP - Old GDP) / Old GDP × 100%",
                "Growth rate = (1100 - 1000) / 1000 × 100% = 10%"
            ],
            "final_answer" => "The GDP growth rate is 10%."
        ),
        Dict(
            "problem_type" => "game_theory",
            "problem_statement" => "In a prisoner's dilemma, what is the Nash equilibrium?", 
            "solution_steps" => [
                "Each player chooses their dominant strategy",
                "Both players defect because it's the best response regardless of opponent's choice"
            ],
            "final_answer" => "The Nash equilibrium is (Defect, Defect)."
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
    
    println("   Created minimal training data: $output_path")
end

"""
    list_lora_gguf_files(directory::String="lora_gguf")

List all GGUF files in the specified directory.

# Arguments
- `directory::String="lora_gguf"`: Directory to search for GGUF files

# Returns
- `Vector{String}`: List of GGUF file paths

# Example
```julia
gguf_files = list_lora_gguf_files()
for file in gguf_files
    println("Found LoRA: \$file")
end
```
"""
function list_lora_gguf_files(directory::String="lora_gguf")
    if !isdir(directory)
        return String[]
    end
    
    gguf_files = String[]
    for file in readdir(directory, join=true)
        if endswith(lowercase(file), ".gguf")
            push!(gguf_files, file)
        end
    end
    
    return gguf_files
end