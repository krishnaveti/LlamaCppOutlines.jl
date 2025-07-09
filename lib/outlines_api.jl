module OutlinesCppAPI

using Libdl

# ─────────────────────────────────────────────────────────────────────────────
# Type Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Opaque pointer types
const outlines_vocabulary = UInt64
const outlines_index = UInt64

# ─────────────────────────────────────────────────────────────────────────────
# Global Variables (set by init!)
# ─────────────────────────────────────────────────────────────────────────────

# DLL handle and function pointers - initialized by init!()
global handle::Ptr{Nothing} = C_NULL

# Core functions
global version_fn::Ptr{Nothing} = C_NULL
global free_string_fn::Ptr{Nothing} = C_NULL

# Schema functions
global regex_from_schema_fn::Ptr{Nothing} = C_NULL

# Vocabulary functions
global create_vocabulary_fn::Ptr{Nothing} = C_NULL
global create_vocabulary_with_token_fn::Ptr{Nothing} = C_NULL
global vocabulary_size_fn::Ptr{Nothing} = C_NULL
global vocabulary_eos_token_id_fn::Ptr{Nothing} = C_NULL
global free_vocabulary_fn::Ptr{Nothing} = C_NULL

# Index functions
global create_index_fn::Ptr{Nothing} = C_NULL
global index_initial_state_fn::Ptr{Nothing} = C_NULL
global index_allowed_tokens_fn::Ptr{Nothing} = C_NULL
global index_next_state_fn::Ptr{Nothing} = C_NULL
global index_is_final_state_fn::Ptr{Nothing} = C_NULL
global free_index_fn::Ptr{Nothing} = C_NULL

# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

"""
    init!(dll_path::String)

Initialize the OutlinesCpp module by loading the DLL and resolving function pointers.
Must be called before using any other functions.

# Arguments
- `dll_path::String`: Path to the outlines_core.dll file

# Example
```julia
OutlinesCpp.init!("path/to/outlines_core.dll")
```
"""
function init!(dll_path::String)
    global handle = dlopen(dll_path)

    # Core functions
    global version_fn = dlsym(handle, :outlines_version)
    global free_string_fn = dlsym(handle, :outlines_free_string)

    # Schema functions
    global regex_from_schema_fn = dlsym(handle, :outlines_regex_from_schema)

    # Vocabulary functions
    global create_vocabulary_fn = dlsym(handle, :outlines_create_vocabulary)
    global create_vocabulary_with_token_fn = dlsym(handle, :outlines_create_vocabulary_with_token)
    global vocabulary_size_fn = dlsym(handle, :outlines_vocabulary_size)
    global vocabulary_eos_token_id_fn = dlsym(handle, :outlines_vocabulary_eos_token_id)
    global free_vocabulary_fn = dlsym(handle, :outlines_free_vocabulary)

    # Index functions
    global create_index_fn = dlsym(handle, :outlines_create_index)
    global index_initial_state_fn = dlsym(handle, :outlines_index_initial_state)
    global index_allowed_tokens_fn = dlsym(handle, :outlines_index_allowed_tokens)
    global index_next_state_fn = dlsym(handle, :outlines_index_next_state)
    global index_is_final_state_fn = dlsym(handle, :outlines_index_is_final_state)
    global free_index_fn = dlsym(handle, :outlines_free_index)

    println("✓ OutlinesCpp module initialized successfully")
end

# ─────────────────────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    version() -> String

Get the Outlines version string.
"""
function version()
    version_ptr = Ref{Ptr{Cchar}}()
    result = ccall(version_fn, Cint, (Ref{Ptr{Cchar}},), version_ptr)

    if result == 0
        version_str = unsafe_string(version_ptr[])
        ccall(free_string_fn, Cvoid, (Ptr{Cchar},), version_ptr[])
        return version_str
    else
        error("Failed to get Outlines version")
    end
end

"""
    free_string(ptr::Ptr{Cchar})

Free a string allocated by Outlines.
"""
function free_string(ptr::Ptr{Cchar})
    ccall(free_string_fn, Cvoid, (Ptr{Cchar},), ptr)
end

# ─────────────────────────────────────────────────────────────────────────────
# Schema Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    regex_from_schema(schema::String) -> String

Generate a regex pattern from a JSON schema.

# Arguments
- `schema::String`: JSON schema string

# Returns
- `String`: Generated regex pattern

# Example
```julia
schema = "{\"type\": \"string\"}"
regex = OutlinesCpp.regex_from_schema(schema)
```
"""
function regex_from_schema(schema::String)
    regex_ptr = Ref{Ptr{Cchar}}()
    result = ccall(regex_from_schema_fn, Cint,
        (Cstring, Ref{Ptr{Cchar}}), schema, regex_ptr)

    if result == 0
        regex_str = unsafe_string(regex_ptr[])
        ccall(free_string_fn, Cvoid, (Ptr{Cchar},), regex_ptr[])
        return regex_str
    else
        error("Failed to generate regex from schema (error code: $result)")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    create_vocabulary(model_name::String) -> outlines_vocabulary

Create a vocabulary from a model name.

# Arguments
- `model_name::String`: HuggingFace model name (e.g., "openai-community/gpt2")

# Returns
- `outlines_vocabulary`: Handle to the vocabulary

# Example
```julia
vocab = OutlinesCpp.create_vocabulary("openai-community/gpt2")
```
"""
function create_vocabulary(model_name::String)
    vocab_handle = Ref{outlines_vocabulary}(0)
    result = ccall(create_vocabulary_fn, Cint,
        (Cstring, Ref{outlines_vocabulary}), model_name, vocab_handle)

    if result == 0
        return vocab_handle[]
    else
        error("Failed to create vocabulary for model '$model_name' (error code: $result)")
    end
end

"""
    create_vocabulary_with_token(model_name::String, token::String) -> outlines_vocabulary

Create a vocabulary from a model name with authentication token.

# Arguments
- `model_name::String`: HuggingFace model name
- `token::String`: HuggingFace authentication token

# Returns
- `outlines_vocabulary`: Handle to the vocabulary
"""
function create_vocabulary_with_token(model_name::String, token::String)
    vocab_handle = Ref{outlines_vocabulary}(0)
    result = ccall(create_vocabulary_with_token_fn, Cint,
        (Cstring, Cstring, Ref{outlines_vocabulary}), model_name, token, vocab_handle)

    if result == 0
        return vocab_handle[]
    else
        error("Failed to create vocabulary for model '$model_name' with token (error code: $result)")
    end
end

"""
    vocabulary_size(vocab::outlines_vocabulary) -> UInt64

Get the size of a vocabulary.

# Arguments
- `vocab::outlines_vocabulary`: Vocabulary handle

# Returns
- `UInt64`: Number of tokens in the vocabulary
"""
function vocabulary_size(vocab::outlines_vocabulary)
    size_out = Ref{UInt64}(0)
    result = ccall(vocabulary_size_fn, Cint,
        (outlines_vocabulary, Ref{UInt64}), vocab, size_out)

    if result == 0
        return size_out[]
    else
        error("Failed to get vocabulary size (error code: $result)")
    end
end

"""
    vocabulary_eos_token_id(vocab::outlines_vocabulary) -> UInt32

Get the EOS (end-of-sequence) token ID from a vocabulary.

# Arguments
- `vocab::outlines_vocabulary`: Vocabulary handle

# Returns
- `UInt32`: EOS token ID
"""
function vocabulary_eos_token_id(vocab::outlines_vocabulary)
    eos_id = Ref{UInt32}(0)
    result = ccall(vocabulary_eos_token_id_fn, Cint,
        (outlines_vocabulary, Ref{UInt32}), vocab, eos_id)

    if result == 0
        return eos_id[]
    else
        error("Failed to get EOS token ID (error code: $result)")
    end
end

"""
    free_vocabulary(vocab::outlines_vocabulary)

Free a vocabulary.

# Arguments
- `vocab::outlines_vocabulary`: Vocabulary handle to free
"""
function free_vocabulary(vocab::outlines_vocabulary)
    ccall(free_vocabulary_fn, Cvoid, (outlines_vocabulary,), vocab)
end

# ─────────────────────────────────────────────────────────────────────────────
# Index Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    create_index(regex::String, vocab::outlines_vocabulary) -> outlines_index

Create an index from a regex pattern and vocabulary.

# Arguments
- `regex::String`: Regex pattern
- `vocab::outlines_vocabulary`: Vocabulary handle

# Returns
- `outlines_index`: Handle to the index

# Example
```julia
index = OutlinesCpp.create_index(regex_pattern, vocab_handle)
```
"""
function create_index(regex::String, vocab::outlines_vocabulary)
    index_handle = Ref{outlines_index}(0)
    result = ccall(create_index_fn, Cint,
        (Cstring, outlines_vocabulary, Ref{outlines_index}), regex, vocab, index_handle)

    if result == 0
        return index_handle[]
    else
        error("Failed to create index (error code: $result)")
    end
end

"""
    index_initial_state(index::outlines_index) -> UInt32

Get the initial state for an index.

# Arguments
- `index::outlines_index`: Index handle

# Returns
- `UInt32`: Initial state ID
"""
function index_initial_state(index::outlines_index)
    initial_state = Ref{UInt32}(0)
    result = ccall(index_initial_state_fn, Cint,
        (outlines_index, Ref{UInt32}), index, initial_state)

    if result == 0
        return initial_state[]
    else
        error("Failed to get initial state (error code: $result)")
    end
end

"""
    index_allowed_tokens(index::outlines_index, state::UInt32) -> Vector{UInt32}

Get allowed tokens for a given state.

# Arguments
- `index::outlines_index`: Index handle
- `state::UInt32`: Current state ID

# Returns
- `Vector{UInt32}`: Array of allowed token IDs
"""
function index_allowed_tokens(index::outlines_index, state::UInt32)
    tokens_ptr = Ref{Ptr{UInt32}}()
    count_ptr = Ref{UInt64}(0)

    result = ccall(index_allowed_tokens_fn, Cint,
        (outlines_index, UInt32, Ref{Ptr{UInt32}}, Ref{UInt64}),
        index, state, tokens_ptr, count_ptr)

    if result == 0 && count_ptr[] > 0
        tokens = unsafe_wrap(Array, tokens_ptr[], (Int(count_ptr[]),))
        return collect(tokens)  # Make a copy
    else
        return UInt32[]
    end
end

"""
    index_next_state(index::outlines_index, state::UInt32, token::UInt32) -> UInt32

Get the next state after consuming a token.

# Arguments
- `index::outlines_index`: Index handle
- `state::UInt32`: Current state ID
- `token::UInt32`: Token ID to consume

# Returns
- `UInt32`: Next state ID
"""
function index_next_state(index::outlines_index, state::UInt32, token::UInt32)
    next_state = Ref{UInt32}(0)
    result = ccall(index_next_state_fn, Cint,
        (outlines_index, UInt32, UInt32, Ref{UInt32}),
        index, state, token, next_state)

    if result == 0
        return next_state[]
    else
        error("Failed to get next state (error code: $result)")
    end
end

"""
    index_is_final_state(index::outlines_index, state::UInt32) -> Bool

Check if a state is a final (accepting) state.

# Arguments
- `index::outlines_index`: Index handle
- `state::UInt32`: State ID to check

# Returns
- `Bool`: True if the state is final
"""
function index_is_final_state(index::outlines_index, state::UInt32)
    is_final = Ref{Bool}(false)
    result = ccall(index_is_final_state_fn, Cint,
        (outlines_index, UInt32, Ref{Bool}), index, state, is_final)

    if result == 0
        return is_final[]
    else
        error("Failed to check if state is final (error code: $result)")
    end
end

"""
    free_index(index::outlines_index)

Free an index.

# Arguments
- `index::outlines_index`: Index handle to free
"""
function free_index(index::outlines_index)
    ccall(free_index_fn, Cvoid, (outlines_index,), index)
end

# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

# Types
export outlines_vocabulary, outlines_index

# Initialization
export init!

# Core functions
export version, free_string

# Schema functions
export regex_from_schema

# Vocabulary functions
export create_vocabulary, create_vocabulary_with_token
export vocabulary_size, vocabulary_eos_token_id, free_vocabulary

# Index functions
export create_index, index_initial_state, index_allowed_tokens
export index_next_state, index_is_final_state, free_index

end # module OutlinesCppAPI