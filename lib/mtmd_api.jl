module MtmdCppAPI

using Libdl

# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

@enum MtmdInputChunkType begin
    MTMD_INPUT_CHUNK_TYPE_TEXT = 0
    MTMD_INPUT_CHUNK_TYPE_IMAGE = 1
    MTMD_INPUT_CHUNK_TYPE_AUDIO = 2
end

# ─────────────────────────────────────────────────────────────────────────────
# Type Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Opaque pointer types
const mtmd_context = Ptr{Cvoid}
const mtmd_bitmap = Ptr{Cvoid}
const mtmd_image_tokens = Ptr{Cvoid}
const mtmd_input_chunk = Ptr{Cvoid}
const mtmd_input_chunks = Ptr{Cvoid}

# LLaMA types
const llama_token = Cint
const llama_pos = Cint

# ─────────────────────────────────────────────────────────────────────────────
# Structs
# ─────────────────────────────────────────────────────────────────────────────

mutable struct mtmd_context_params
    use_gpu::Bool
    print_timings::Bool
    n_threads::Cint
    verbosity::Cint  # ggml_log_level
    image_marker::Cstring  # deprecated
    media_marker::Cstring
end

mutable struct mtmd_input_text
    text::Cstring
    add_special::Bool
    parse_special::Bool
end

# ─────────────────────────────────────────────────────────────────────────────
# Global Variables (set by init!)
# ─────────────────────────────────────────────────────────────────────────────

# DLL handle and function pointers - initialized by init!()
global handle::Ptr{Nothing} = C_NULL

# Core context functions
global mtmd_ctx_params_default_fn::Ptr{Nothing} = C_NULL
global mtmd_init_from_file_fn::Ptr{Nothing} = C_NULL
global mtmd_free_fn::Ptr{Nothing} = C_NULL
global mtmd_default_marker_fn::Ptr{Nothing} = C_NULL

# Capability check functions
global mtmd_support_vision_fn::Ptr{Nothing} = C_NULL
global mtmd_support_audio_fn::Ptr{Nothing} = C_NULL
global mtmd_get_audio_bitrate_fn::Ptr{Nothing} = C_NULL
global mtmd_decode_use_non_causal_fn::Ptr{Nothing} = C_NULL
global mtmd_decode_use_mrope_fn::Ptr{Nothing} = C_NULL

# Bitmap functions
global mtmd_bitmap_init_fn::Ptr{Nothing} = C_NULL
global mtmd_bitmap_init_from_audio_fn::Ptr{Nothing} = C_NULL
global mtmd_bitmap_free_fn::Ptr{Nothing} = C_NULL
global mtmd_bitmap_get_nx_fn::Ptr{Nothing} = C_NULL
global mtmd_bitmap_get_ny_fn::Ptr{Nothing} = C_NULL
global mtmd_bitmap_get_data_fn::Ptr{Nothing} = C_NULL
global mtmd_bitmap_get_n_bytes_fn::Ptr{Nothing} = C_NULL
global mtmd_bitmap_is_audio_fn::Ptr{Nothing} = C_NULL
global mtmd_bitmap_get_id_fn::Ptr{Nothing} = C_NULL
global mtmd_bitmap_set_id_fn::Ptr{Nothing} = C_NULL

# Input chunks functions
global mtmd_input_chunks_init_fn::Ptr{Nothing} = C_NULL
global mtmd_input_chunks_size_fn::Ptr{Nothing} = C_NULL
global mtmd_input_chunks_get_fn::Ptr{Nothing} = C_NULL
global mtmd_input_chunks_free_fn::Ptr{Nothing} = C_NULL

# Input chunk functions
global mtmd_input_chunk_get_type_fn::Ptr{Nothing} = C_NULL
global mtmd_input_chunk_get_tokens_text_fn::Ptr{Nothing} = C_NULL
global mtmd_input_chunk_get_tokens_image_fn::Ptr{Nothing} = C_NULL
global mtmd_input_chunk_get_n_tokens_fn::Ptr{Nothing} = C_NULL
global mtmd_input_chunk_get_id_fn::Ptr{Nothing} = C_NULL
global mtmd_input_chunk_get_n_pos_fn::Ptr{Nothing} = C_NULL
global mtmd_input_chunk_copy_fn::Ptr{Nothing} = C_NULL
global mtmd_input_chunk_free_fn::Ptr{Nothing} = C_NULL

# Image tokens functions
global mtmd_image_tokens_get_n_tokens_fn::Ptr{Nothing} = C_NULL
global mtmd_image_tokens_get_nx_fn::Ptr{Nothing} = C_NULL
global mtmd_image_tokens_get_ny_fn::Ptr{Nothing} = C_NULL
global mtmd_image_tokens_get_id_fn::Ptr{Nothing} = C_NULL
global mtmd_image_tokens_get_n_pos_fn::Ptr{Nothing} = C_NULL

# Processing functions
global mtmd_tokenize_fn::Ptr{Nothing} = C_NULL
global mtmd_encode_fn::Ptr{Nothing} = C_NULL
global mtmd_encode_chunk_fn::Ptr{Nothing} = C_NULL
global mtmd_get_output_embd_fn::Ptr{Nothing} = C_NULL

# Helper functions
global mtmd_helper_eval_chunks_fn::Ptr{Nothing} = C_NULL
global mtmd_helper_get_n_tokens_fn::Ptr{Nothing} = C_NULL
global mtmd_helper_get_n_pos_fn::Ptr{Nothing} = C_NULL
global mtmd_helper_decode_image_chunk_fn::Ptr{Nothing} = C_NULL

# Test function
global mtmd_test_create_input_chunks_fn::Ptr{Nothing} = C_NULL

# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────

"""
    init!(dll_path::String)

Initialize the MtmdCpp module by loading the DLL and resolving function pointers.
Must be called before using any other functions.

# Arguments
- `dll_path::String`: Path to the mtmd.dll file

# Example
```julia
MtmdCpp.init!("path/to/mtmd.dll")
```
"""
function init!(dll_path::String)
    global handle = dlopen(dll_path)

    # Core context functions
    global mtmd_ctx_params_default_fn = dlsym(handle, :mtmd_context_params_default)
    global mtmd_init_from_file_fn = dlsym(handle, :mtmd_init_from_file)
    global mtmd_free_fn = dlsym(handle, :mtmd_free)
    global mtmd_default_marker_fn = dlsym(handle, :mtmd_default_marker)

    # Capability check functions
    global mtmd_support_vision_fn = dlsym(handle, :mtmd_support_vision)
    global mtmd_support_audio_fn = dlsym(handle, :mtmd_support_audio)
    global mtmd_get_audio_bitrate_fn = dlsym(handle, :mtmd_get_audio_bitrate)
    global mtmd_decode_use_non_causal_fn = dlsym(handle, :mtmd_decode_use_non_causal)
    global mtmd_decode_use_mrope_fn = dlsym(handle, :mtmd_decode_use_mrope)

    # Bitmap functions
    global mtmd_bitmap_init_fn = dlsym(handle, :mtmd_bitmap_init)
    global mtmd_bitmap_init_from_audio_fn = dlsym(handle, :mtmd_bitmap_init_from_audio)
    global mtmd_bitmap_free_fn = dlsym(handle, :mtmd_bitmap_free)
    global mtmd_bitmap_get_nx_fn = dlsym(handle, :mtmd_bitmap_get_nx)
    global mtmd_bitmap_get_ny_fn = dlsym(handle, :mtmd_bitmap_get_ny)
    global mtmd_bitmap_get_data_fn = dlsym(handle, :mtmd_bitmap_get_data)
    global mtmd_bitmap_get_n_bytes_fn = dlsym(handle, :mtmd_bitmap_get_n_bytes)
    global mtmd_bitmap_is_audio_fn = dlsym(handle, :mtmd_bitmap_is_audio)
    global mtmd_bitmap_get_id_fn = dlsym(handle, :mtmd_bitmap_get_id)
    global mtmd_bitmap_set_id_fn = dlsym(handle, :mtmd_bitmap_set_id)

    # Input chunks functions
    global mtmd_input_chunks_init_fn = dlsym(handle, :mtmd_input_chunks_init)
    global mtmd_input_chunks_size_fn = dlsym(handle, :mtmd_input_chunks_size)
    global mtmd_input_chunks_get_fn = dlsym(handle, :mtmd_input_chunks_get)
    global mtmd_input_chunks_free_fn = dlsym(handle, :mtmd_input_chunks_free)

    # Input chunk functions
    global mtmd_input_chunk_get_type_fn = dlsym(handle, :mtmd_input_chunk_get_type)
    global mtmd_input_chunk_get_tokens_text_fn = dlsym(handle, :mtmd_input_chunk_get_tokens_text)
    global mtmd_input_chunk_get_tokens_image_fn = dlsym(handle, :mtmd_input_chunk_get_tokens_image)
    global mtmd_input_chunk_get_n_tokens_fn = dlsym(handle, :mtmd_input_chunk_get_n_tokens)
    global mtmd_input_chunk_get_id_fn = dlsym(handle, :mtmd_input_chunk_get_id)
    global mtmd_input_chunk_get_n_pos_fn = dlsym(handle, :mtmd_input_chunk_get_n_pos)
    global mtmd_input_chunk_copy_fn = dlsym(handle, :mtmd_input_chunk_copy)
    global mtmd_input_chunk_free_fn = dlsym(handle, :mtmd_input_chunk_free)

    # Image tokens functions
    global mtmd_image_tokens_get_n_tokens_fn = dlsym(handle, :mtmd_image_tokens_get_n_tokens)
    global mtmd_image_tokens_get_nx_fn = dlsym(handle, :mtmd_image_tokens_get_nx)
    global mtmd_image_tokens_get_ny_fn = dlsym(handle, :mtmd_image_tokens_get_ny)
    global mtmd_image_tokens_get_id_fn = dlsym(handle, :mtmd_image_tokens_get_id)
    global mtmd_image_tokens_get_n_pos_fn = dlsym(handle, :mtmd_image_tokens_get_n_pos)

    # Processing functions
    global mtmd_tokenize_fn = dlsym(handle, :mtmd_tokenize)
    global mtmd_encode_fn = dlsym(handle, :mtmd_encode)
    global mtmd_encode_chunk_fn = dlsym(handle, :mtmd_encode_chunk)
    global mtmd_get_output_embd_fn = dlsym(handle, :mtmd_get_output_embd)

    # Helper functions
    global mtmd_helper_eval_chunks_fn = dlsym(handle, :mtmd_helper_eval_chunks)
    global mtmd_helper_get_n_tokens_fn = dlsym(handle, :mtmd_helper_get_n_tokens)
    global mtmd_helper_get_n_pos_fn = dlsym(handle, :mtmd_helper_get_n_pos)
    global mtmd_helper_decode_image_chunk_fn = dlsym(handle, :mtmd_helper_decode_image_chunk)

    # Test function
    global mtmd_test_create_input_chunks_fn = dlsym(handle, :mtmd_test_create_input_chunks)

    println("✓ MtmdCpp module initialized successfully")
end

# ─────────────────────────────────────────────────────────────────────────────
# Core Context Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    default_marker() -> String

Get the default media marker string (e.g., "<__media__>").
"""
function default_marker()
    cstr = ccall(mtmd_default_marker_fn, Cstring, ())
    return unsafe_string(cstr)
end

"""
    context_params_default() -> mtmd_context_params

Get default context parameters.
"""
function context_params_default()
    return ccall(mtmd_ctx_params_default_fn, mtmd_context_params, ())
end

"""
    init_from_file(mmproj_fname::String, text_model::Ptr{Cvoid}, params::mtmd_context_params) -> mtmd_context

Initialize MTMD context from file. Returns C_NULL on failure.
"""
function init_from_file(mmproj_fname::String, text_model::Ptr{Cvoid}, params::mtmd_context_params)
    return ccall(mtmd_init_from_file_fn, mtmd_context,
        (Cstring, Ptr{Cvoid}, mtmd_context_params),
        mmproj_fname, text_model, params)
end

"""
    free_context(ctx::mtmd_context)

Free MTMD context.
"""
function free_context(ctx::mtmd_context)
    ccall(mtmd_free_fn, Cvoid, (mtmd_context,), ctx)
end

"""
    decode_use_non_causal(ctx::mtmd_context) -> Bool

Check whether we need to set non-causal mask before llama_decode.
"""
function decode_use_non_causal(ctx::mtmd_context)
    return ccall(mtmd_decode_use_non_causal_fn, Bool, (mtmd_context,), ctx)
end

"""
    decode_use_mrope(ctx::mtmd_context) -> Bool

Check whether the current model uses M-RoPE for llama_decode.
"""
function decode_use_mrope(ctx::mtmd_context)
    return ccall(mtmd_decode_use_mrope_fn, Bool, (mtmd_context,), ctx)
end

"""
    support_vision(ctx::mtmd_context) -> Bool

Check if the model supports vision input.
"""
function support_vision(ctx::mtmd_context)
    return ccall(mtmd_support_vision_fn, Bool, (mtmd_context,), ctx)
end

"""
    support_audio(ctx::mtmd_context) -> Bool

Check if the model supports audio input.
"""
function support_audio(ctx::mtmd_context)
    return ccall(mtmd_support_audio_fn, Bool, (mtmd_context,), ctx)
end

"""
    get_audio_bitrate(ctx::mtmd_context) -> Int

Get audio bitrate in Hz (e.g., 16000 for Whisper). Returns -1 if audio is not supported.
"""
function get_audio_bitrate(ctx::mtmd_context)
    return ccall(mtmd_get_audio_bitrate_fn, Cint, (mtmd_context,), ctx)
end

# ─────────────────────────────────────────────────────────────────────────────
# Bitmap Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    bitmap_init(nx::UInt32, ny::UInt32, data::Vector{UInt8}) -> mtmd_bitmap

Create a bitmap from RGB image data.
Length of data must be nx * ny * 3 in RGBRGBRGB... format.
"""
function bitmap_init(nx::UInt32, ny::UInt32, data::Vector{UInt8})
    return ccall(mtmd_bitmap_init_fn, mtmd_bitmap,
        (UInt32, UInt32, Ptr{UInt8}),
        nx, ny, pointer(data))
end

"""
    bitmap_init_from_audio(n_samples::UInt, data::Vector{Float32}) -> mtmd_bitmap

Create a bitmap from audio data.
Length of data must be n_samples in PCM F32 format.
"""
function bitmap_init_from_audio(n_samples::UInt, data::Vector{Float32})
    return ccall(mtmd_bitmap_init_from_audio_fn, mtmd_bitmap,
        (Csize_t, Ptr{Cfloat}),
        n_samples, pointer(data))
end

"""
    (bitmap::mtmd_bitmap)

Free a bitmap.
"""
function bitmap_free(bitmap::mtmd_bitmap)
    ccall(mtmd_bitmap_free_fn, Cvoid, (mtmd_bitmap,), bitmap)
end

"""
    bitmap_get_nx(bitmap::mtmd_bitmap) -> UInt32

Get bitmap width.
"""
function bitmap_get_nx(bitmap::mtmd_bitmap)
    return ccall(mtmd_bitmap_get_nx_fn, UInt32, (mtmd_bitmap,), bitmap)
end

"""
    bitmap_get_ny(bitmap::mtmd_bitmap) -> UInt32

Get bitmap height.
"""
function bitmap_get_ny(bitmap::mtmd_bitmap)
    return ccall(mtmd_bitmap_get_ny_fn, UInt32, (mtmd_bitmap,), bitmap)
end

"""
    bitmap_get_data(bitmap::mtmd_bitmap) -> Ptr{UInt8}

Get bitmap data pointer.
"""
function bitmap_get_data(bitmap::mtmd_bitmap)
    return ccall(mtmd_bitmap_get_data_fn, Ptr{UInt8}, (mtmd_bitmap,), bitmap)
end

"""
    bitmap_get_n_bytes(bitmap::mtmd_bitmap) -> UInt

Get bitmap data size in bytes.
"""
function bitmap_get_n_bytes(bitmap::mtmd_bitmap)
    return ccall(mtmd_bitmap_get_n_bytes_fn, Csize_t, (mtmd_bitmap,), bitmap)
end

"""
    bitmap_is_audio(bitmap::mtmd_bitmap) -> Bool

Check if bitmap contains audio data.
"""
function bitmap_is_audio(bitmap::mtmd_bitmap)
    return ccall(mtmd_bitmap_is_audio_fn, Bool, (mtmd_bitmap,), bitmap)
end

"""
    bitmap_get_id(bitmap::mtmd_bitmap) -> String

Get bitmap ID (optional, useful for KV cache tracking).
"""
function bitmap_get_id(bitmap::mtmd_bitmap)
    cstr = ccall(mtmd_bitmap_get_id_fn, Cstring, (mtmd_bitmap,), bitmap)
    return cstr == C_NULL ? "" : unsafe_string(cstr)
end

"""
    bitmap_set_id(bitmap::mtmd_bitmap, id::String)

Set bitmap ID.
"""
function bitmap_set_id(bitmap::mtmd_bitmap, id::String)
    ccall(mtmd_bitmap_set_id_fn, Cvoid, (mtmd_bitmap, Cstring), bitmap, id)
end

# ─────────────────────────────────────────────────────────────────────────────
# Input Chunks Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    input_chunks_init() -> mtmd_input_chunks

Initialize input chunks container.
"""
function input_chunks_init()
    return ccall(mtmd_input_chunks_init_fn, mtmd_input_chunks, ())
end

"""
    input_chunks_size(chunks::mtmd_input_chunks) -> UInt

Get number of chunks.
"""
function input_chunks_size(chunks::mtmd_input_chunks)
    return ccall(mtmd_input_chunks_size_fn, Csize_t, (mtmd_input_chunks,), chunks)
end

"""
    input_chunks_get(chunks::mtmd_input_chunks, idx::UInt) -> mtmd_input_chunk

Get chunk at index.
"""
function input_chunks_get(chunks::mtmd_input_chunks, idx::UInt)
    return ccall(mtmd_input_chunks_get_fn, mtmd_input_chunk,
        (mtmd_input_chunks, Csize_t), chunks, idx)
end

"""
    input_chunks_free(chunks::mtmd_input_chunks)

Free input chunks.
"""
function input_chunks_free(chunks::mtmd_input_chunks)
    ccall(mtmd_input_chunks_free_fn, Cvoid, (mtmd_input_chunks,), chunks)
end

# ─────────────────────────────────────────────────────────────────────────────
# Input Chunk Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    input_chunk_get_type(chunk::mtmd_input_chunk) -> MtmdInputChunkType

Get chunk type (text, image, or audio).
"""
function input_chunk_get_type(chunk::mtmd_input_chunk)
    code = ccall(mtmd_input_chunk_get_type_fn, Cint, (mtmd_input_chunk,), chunk)
    return MtmdInputChunkType(code)
end

"""
    input_chunk_get_tokens_text(chunk::mtmd_input_chunk) -> Vector{llama_token}

Get text tokens from a text chunk.
"""
function input_chunk_get_tokens_text(chunk::mtmd_input_chunk)
    n_tokens_ref = Ref{Csize_t}(0)
    tokens_ptr = ccall(mtmd_input_chunk_get_tokens_text_fn, Ptr{llama_token},
        (mtmd_input_chunk, Ref{Csize_t}), chunk, n_tokens_ref)

    n_tokens = n_tokens_ref[]
    if tokens_ptr == C_NULL || n_tokens == 0
        return llama_token[]
    end

    return unsafe_wrap(Array, tokens_ptr, n_tokens; own=false) |> copy
end

"""
    input_chunk_get_tokens_image(chunk::mtmd_input_chunk) -> mtmd_image_tokens

Get image tokens from an image chunk.
"""
function input_chunk_get_tokens_image(chunk::mtmd_input_chunk)
    return ccall(mtmd_input_chunk_get_tokens_image_fn, mtmd_image_tokens, (mtmd_input_chunk,), chunk)
end

"""
    input_chunk_get_n_tokens(chunk::mtmd_input_chunk) -> UInt

Get number of tokens in chunk.
"""
function input_chunk_get_n_tokens(chunk::mtmd_input_chunk)
    return ccall(mtmd_input_chunk_get_n_tokens_fn, Csize_t, (mtmd_input_chunk,), chunk)
end

"""
    input_chunk_get_id(chunk::mtmd_input_chunk) -> String

Get chunk ID (returns empty string for text chunks).
"""
function input_chunk_get_id(chunk::mtmd_input_chunk)
    cstr = ccall(mtmd_input_chunk_get_id_fn, Cstring, (mtmd_input_chunk,), chunk)
    return cstr == C_NULL ? "" : unsafe_string(cstr)
end

"""
    input_chunk_get_n_pos(chunk::mtmd_input_chunk) -> llama_pos

Get number of temporal positions (always 1 for M-RoPE, n_tokens otherwise).
"""
function input_chunk_get_n_pos(chunk::mtmd_input_chunk)
    return ccall(mtmd_input_chunk_get_n_pos_fn, llama_pos, (mtmd_input_chunk,), chunk)
end

"""
    input_chunk_copy(chunk::mtmd_input_chunk) -> mtmd_input_chunk

Copy a chunk for custom KV cache management. Remember to free when done.
"""
function input_chunk_copy(chunk::mtmd_input_chunk)
    return ccall(mtmd_input_chunk_copy_fn, mtmd_input_chunk, (mtmd_input_chunk,), chunk)
end

"""
    input_chunk_free(chunk::mtmd_input_chunk)

Free a copied chunk.
"""
function input_chunk_free(chunk::mtmd_input_chunk)
    ccall(mtmd_input_chunk_free_fn, Cvoid, (mtmd_input_chunk,), chunk)
end

# ─────────────────────────────────────────────────────────────────────────────
# Image Tokens Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    image_tokens_get_n_tokens(image_tokens::mtmd_image_tokens) -> UInt

Get number of tokens in image tokens (will be deprecated).
"""
function image_tokens_get_n_tokens(image_tokens::mtmd_image_tokens)
    return ccall(mtmd_image_tokens_get_n_tokens_fn, Csize_t, (mtmd_image_tokens,), image_tokens)
end

"""
    image_tokens_get_nx(image_tokens::mtmd_image_tokens) -> UInt

Get image width in patches.
"""
function image_tokens_get_nx(image_tokens::mtmd_image_tokens)
    return ccall(mtmd_image_tokens_get_nx_fn, Csize_t, (mtmd_image_tokens,), image_tokens)
end

"""
    image_tokens_get_ny(image_tokens::mtmd_image_tokens) -> UInt

Get image height in patches.
"""
function image_tokens_get_ny(image_tokens::mtmd_image_tokens)
    return ccall(mtmd_image_tokens_get_ny_fn, Csize_t, (mtmd_image_tokens,), image_tokens)
end

"""
    image_tokens_get_id(image_tokens::mtmd_image_tokens) -> String

Get image tokens ID (will be deprecated).
"""
function image_tokens_get_id(image_tokens::mtmd_image_tokens)
    cstr = ccall(mtmd_image_tokens_get_id_fn, Cstring, (mtmd_image_tokens,), image_tokens)
    return cstr == C_NULL ? "" : unsafe_string(cstr)
end

"""
    image_tokens_get_n_pos(image_tokens::mtmd_image_tokens) -> llama_pos

Get number of temporal positions (will be deprecated).
"""
function image_tokens_get_n_pos(image_tokens::mtmd_image_tokens)
    return ccall(mtmd_image_tokens_get_n_pos_fn, llama_pos, (mtmd_image_tokens,), image_tokens)
end

# ─────────────────────────────────────────────────────────────────────────────
# Processing Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    tokenize(ctx::mtmd_context, output::mtmd_input_chunks, text::mtmd_input_text, 
             bitmaps::Vector{mtmd_bitmap}) -> Int32

Tokenize multimodal input into chunks.

Returns:
- 0 on success
- 1 on number of bitmaps not matching the number of markers
- 2 on image preprocessing error
"""
function tokenize(ctx::mtmd_context, output::mtmd_input_chunks, text::mtmd_input_text,
    bitmaps::Vector{mtmd_bitmap})
    n_bitmaps = length(bitmaps)
    bitmap_ptrs = [bitmap for bitmap in bitmaps]

    return ccall(mtmd_tokenize_fn, Int32,
        (mtmd_context, mtmd_input_chunks, Ref{mtmd_input_text}, Ptr{mtmd_bitmap}, Csize_t),
        ctx, output, Ref(text), pointer(bitmap_ptrs), Csize_t(n_bitmaps))
end

"""
    encode(ctx::mtmd_context, image_tokens::mtmd_image_tokens) -> Int32

Encode image tokens (deprecated). Returns 0 on success.
"""
function encode(ctx::mtmd_context, image_tokens::mtmd_image_tokens)
    return ccall(mtmd_encode_fn, Int32, (mtmd_context, mtmd_image_tokens), ctx, image_tokens)
end

"""
    encode_chunk(ctx::mtmd_context, chunk::mtmd_input_chunk) -> Int32

Encode a multimodal chunk. Returns 0 on success.
"""
function encode_chunk(ctx::mtmd_context, chunk::mtmd_input_chunk)
    return ccall(mtmd_encode_chunk_fn, Int32, (mtmd_context, mtmd_input_chunk), ctx, chunk)
end

"""
    get_output_embd(ctx::mtmd_context) -> Ptr{Cfloat}

Get output embeddings from the last encode pass.
Reading size (bytes) = llama_model_n_embd(model) * chunk_n_tokens * sizeof(float).
"""
function get_output_embd(ctx::mtmd_context)
    return ccall(mtmd_get_output_embd_fn, Ptr{Cfloat}, (mtmd_context,), ctx)
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper Decoding Functions 
# ─────────────────────────────────────────────────────────────────────────────

"""
    helper_eval_chunks(mm_ctx, llama_ctx, chunks, n_past, seq_id, n_batch, logits_last) -> (success, new_n_past)

The COMPLETE multimodal processing function that handles both text and image chunks properly.
"""
function helper_eval_chunks(mm_ctx::mtmd_context, llama_ctx::Ptr{Cvoid},
    chunks::mtmd_input_chunks, n_past::Integer,
    seq_id::Integer=Int32(0), n_batch::Integer=Int32(4096),
    logits_last::Bool=true)
    new_n_past_ref = Ref{Int32}(0)

    result = ccall(mtmd_helper_eval_chunks_fn, Int32,
        (mtmd_context, Ptr{Cvoid}, mtmd_input_chunks, Int32, Int32, Int32, Bool, Ref{Int32}),
        mm_ctx, llama_ctx, chunks, n_past, seq_id, n_batch, logits_last, new_n_past_ref)

    return (result == 0, new_n_past_ref[])
end

"""
    helper_get_n_tokens(chunks) -> Int

Get total number of tokens from chunks (for KV cache tracking).
"""
function helper_get_n_tokens(chunks::mtmd_input_chunks)
    return Int(ccall(mtmd_helper_get_n_tokens_fn, Csize_t, (mtmd_input_chunks,), chunks))
end

"""
    helper_get_n_pos(chunks) -> Int32

Get total position count (different from n_tokens for M-RoPE models).
"""
function helper_get_n_pos(chunks::mtmd_input_chunks)
    return ccall(mtmd_helper_get_n_pos_fn, Int32, (mtmd_input_chunks,), chunks)
end

# ─────────────────────────────────────────────────────────────────────────────
# Test Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    test_create_input_chunks() -> mtmd_input_chunks

Test function for creating input chunks (used in test-mtmd-c-api.c).
"""
function test_create_input_chunks()
    return ccall(mtmd_test_create_input_chunks_fn, mtmd_input_chunks, ())
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    create_input_text(text::String; add_special::Bool=true, parse_special::Bool=false) -> mtmd_input_text

Create an mtmd_input_text structure from a string.
"""
function create_input_text(text::String; add_special::Bool=true, parse_special::Bool=false)
    return mtmd_input_text(pointer(text), add_special, parse_special)
end

"""
    tokenize_simple(ctx::mtmd_context, text::String, bitmaps::Vector{mtmd_bitmap}; 
                     add_special::Bool=true, parse_special::Bool=false) -> Tuple{mtmd_input_chunks, Int32}

Convenience function for tokenizing text and bitmaps.
Returns (chunks, status_code).
"""
function tokenize_simple(ctx::mtmd_context, text::String, bitmaps::Vector{mtmd_bitmap};
    add_special::Bool=true, parse_special::Bool=false)
    chunks = input_chunks_init()
    text_input = create_input_text(text; add_special=add_special, parse_special=parse_special)
    status = tokenize(ctx, chunks, text_input, bitmaps)
    return chunks, status
end

"""
    encode_chunk_success(ctx::mtmd_context, chunk::mtmd_input_chunk) -> Bool

Encode a chunk and return success status as Bool.
"""
function encode_chunk_success(ctx::mtmd_context, chunk::mtmd_input_chunk)
    return encode_chunk(ctx, chunk) == 0
end

# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

# Types and enums
export MtmdInputChunkType
export MTMD_INPUT_CHUNK_TYPE_TEXT, MTMD_INPUT_CHUNK_TYPE_IMAGE, MTMD_INPUT_CHUNK_TYPE_AUDIO
export mtmd_context, mtmd_bitmap, mtmd_image_tokens, mtmd_input_chunk, mtmd_input_chunks
export mtmd_context_params, mtmd_input_text
export llama_token, llama_pos

# Initialization
export init!

# Core context functions
export default_marker, context_params_default, init_from_file, free_context
export decode_use_non_causal, decode_use_mrope, support_vision, support_audio, get_audio_bitrate

# Bitmap functions
export bitmap_init, bitmap_init_from_audio, bitmap_free
export bitmap_get_nx, bitmap_get_ny, bitmap_get_data, bitmap_get_n_bytes
export bitmap_is_audio, bitmap_get_id, bitmap_set_id

# Input chunks functions
export input_chunks_init, input_chunks_size, input_chunks_get, input_chunks_free

# Input chunk functions
export input_chunk_get_type, input_chunk_get_tokens_text, input_chunk_get_tokens_image
export input_chunk_get_n_tokens, input_chunk_get_id, input_chunk_get_n_pos
export input_chunk_copy, input_chunk_free

# Image tokens functions
export image_tokens_get_n_tokens, image_tokens_get_nx, image_tokens_get_ny
export image_tokens_get_id, image_tokens_get_n_pos

# Processing functions
export tokenize, encode, encode_chunk, get_output_embd

# Test functions
export test_create_input_chunks

# Helper functions
export create_input_text, tokenize_simple, encode_chunk_success

end # module MtmdCpp