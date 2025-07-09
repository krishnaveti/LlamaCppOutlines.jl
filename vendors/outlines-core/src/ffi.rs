// src/ffi.rs - Complete FFI interface for outlines-core Julia integration
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};
use crate::prelude::*;

// Error codes for FFI
pub const SUCCESS: c_int = 0;
pub const ERROR_NULL_POINTER: c_int = -1;
pub const ERROR_INVALID_UTF8: c_int = -2;
pub const ERROR_STRING_CONVERSION: c_int = -3;
pub const ERROR_SCHEMA_PARSING: c_int = -4;
pub const ERROR_VOCABULARY_CREATION: c_int = -5;
pub const ERROR_INDEX_CREATION: c_int = -6;
pub const ERROR_REGEX_COMPILATION: c_int = -7;
pub const ERROR_INVALID_STATE: c_int = -8;
pub const ERROR_INVALID_TOKEN: c_int = -9;
pub const ERROR_GUIDE_CREATION: c_int = -10;

// Opaque handle types for Julia
pub type VocabularyHandle = usize;
pub type IndexHandle = usize;
pub type GuideHandle = usize;

//=============================================================================
// JSON Schema and Regex Functions
//=============================================================================

/// Build regex pattern from JSON schema string
#[no_mangle]
pub extern "C" fn outlines_regex_from_schema(
    schema_ptr: *const c_char,
    regex_out: *mut *mut c_char
) -> c_int {
    if schema_ptr.is_null() || regex_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let schema_cstr = CStr::from_ptr(schema_ptr);
        let schema_str = match schema_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return ERROR_INVALID_UTF8,
        };
        
        match crate::json_schema::regex_from_str(schema_str, None, None) {
            Ok(regex) => {
                let c_string = match CString::new(regex) {
                    Ok(cs) => cs,
                    Err(_) => return ERROR_STRING_CONVERSION,
                };
                *regex_out = c_string.into_raw();
                SUCCESS
            }
            Err(_) => ERROR_SCHEMA_PARSING
        }
    }
}

/// Build regex pattern from JSON schema with maximum recursion depth
#[no_mangle]
pub extern "C" fn outlines_regex_from_schema_with_depth(
    schema_ptr: *const c_char,
    max_depth: c_uint,
    regex_out: *mut *mut c_char
) -> c_int {
    if schema_ptr.is_null() || regex_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let schema_cstr = CStr::from_ptr(schema_ptr);
        let schema_str = match schema_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return ERROR_INVALID_UTF8,
        };
        
        match crate::json_schema::regex_from_str(schema_str, None, Some(max_depth as usize)) {
            Ok(regex) => {
                let c_string = match CString::new(regex) {
                    Ok(cs) => cs,
                    Err(_) => return ERROR_STRING_CONVERSION,
                };
                *regex_out = c_string.into_raw();
                SUCCESS
            }
            Err(_) => ERROR_SCHEMA_PARSING
        }
    }
}

//=============================================================================
// Vocabulary Functions
//=============================================================================

/// Create vocabulary from pretrained model with optional authentication token
#[no_mangle]
pub extern "C" fn outlines_create_vocabulary(
    model_name_ptr: *const c_char,
    vocab_handle: *mut VocabularyHandle
) -> c_int {
    outlines_create_vocabulary_with_token(model_name_ptr, std::ptr::null(), vocab_handle)
}

/// Create vocabulary from pretrained model with authentication token
#[no_mangle]
pub extern "C" fn outlines_create_vocabulary_with_token(
    model_name_ptr: *const c_char,
    token_ptr: *const c_char,
    vocab_handle: *mut VocabularyHandle
) -> c_int {
    if model_name_ptr.is_null() || vocab_handle.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let model_name_cstr = CStr::from_ptr(model_name_ptr);
        let model_name = match model_name_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return ERROR_INVALID_UTF8,
        };
        
        // Handle optional token - use correct field name "token"
        let parameters = if token_ptr.is_null() {
            None
        } else {
            let token_cstr = CStr::from_ptr(token_ptr);
            match token_cstr.to_str() {
                Ok(token) => {
                    if token.is_empty() {
                        None
                    } else {
                        let mut params = crate::prelude::FromPretrainedParameters::default();
                        params.token = Some(token.to_string()); // ✅ Correct field name
                        Some(params)
                    }
                }
                Err(_) => return ERROR_INVALID_UTF8,
            }
        };
        
        match Vocabulary::from_pretrained(model_name, parameters) {
            Ok(vocab) => {
                let boxed_vocab = Box::new(vocab);
                *vocab_handle = Box::into_raw(boxed_vocab) as VocabularyHandle;
                SUCCESS
            }
            Err(_) => ERROR_VOCABULARY_CREATION
        }
    }
}

/// Create vocabulary from pretrained model with revision and optional authentication token
#[no_mangle]
pub extern "C" fn outlines_create_vocabulary_with_revision(
    model_name_ptr: *const c_char,
    revision_ptr: *const c_char,
    vocab_handle: *mut VocabularyHandle
) -> c_int {
    outlines_create_vocabulary_with_revision_and_token(
        model_name_ptr, 
        revision_ptr, 
        std::ptr::null(), 
        vocab_handle
    )
}

/// Create vocabulary from pretrained model with revision and authentication token
#[no_mangle]
pub extern "C" fn outlines_create_vocabulary_with_revision_and_token(
    model_name_ptr: *const c_char,
    revision_ptr: *const c_char,
    token_ptr: *const c_char,
    vocab_handle: *mut VocabularyHandle
) -> c_int {
    if model_name_ptr.is_null() || vocab_handle.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let model_name_cstr = CStr::from_ptr(model_name_ptr);
        let model_name = match model_name_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return ERROR_INVALID_UTF8,
        };
        
        // Build parameters with both revision and token
        let parameters = {
            let mut params = crate::prelude::FromPretrainedParameters::default();
            
            // Set revision if provided and not empty
            if !revision_ptr.is_null() {
                let revision_cstr = CStr::from_ptr(revision_ptr);
                match revision_cstr.to_str() {
                    Ok(revision) => {
                        if !revision.is_empty() {
                            params.revision = revision.to_string();
                        }
                    }
                    Err(_) => return ERROR_INVALID_UTF8,
                }
            }
            
            // Set auth token if provided and not empty - use correct field name
            if !token_ptr.is_null() {
                let token_cstr = CStr::from_ptr(token_ptr);
                match token_cstr.to_str() {
                    Ok(token) => {
                        if !token.is_empty() {
                            params.token = Some(token.to_string()); // ✅ Correct field name
                        }
                    }
                    Err(_) => return ERROR_INVALID_UTF8,
                }
            }
            
            Some(params)
        };
        
        match Vocabulary::from_pretrained(model_name, parameters) {
            Ok(vocab) => {
                let boxed_vocab = Box::new(vocab);
                *vocab_handle = Box::into_raw(boxed_vocab) as VocabularyHandle;
                SUCCESS
            }
            Err(_) => ERROR_VOCABULARY_CREATION
        }
    }
}

/// Get vocabulary size
#[no_mangle]
pub extern "C" fn outlines_vocabulary_size(
    vocab_handle: VocabularyHandle,
    size_out: *mut usize
) -> c_int {
    if vocab_handle == 0 || size_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let vocab = &*(vocab_handle as *const Vocabulary);
        *size_out = vocab.len();
        SUCCESS
    }
}

/// Get EOS token ID from vocabulary
#[no_mangle]
pub extern "C" fn outlines_vocabulary_eos_token_id(
    vocab_handle: VocabularyHandle,
    eos_token_id_out: *mut u32
) -> c_int {
    if vocab_handle == 0 || eos_token_id_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let vocab = &*(vocab_handle as *const Vocabulary);
        *eos_token_id_out = vocab.eos_token_id();
        SUCCESS
    }
}

/// Get token IDs for a given token string
#[no_mangle]
pub extern "C" fn outlines_vocabulary_get_token_ids(
    vocab_handle: VocabularyHandle,
    token_ptr: *const c_char,
    token_ids_out: *mut *mut u32,
    count_out: *mut usize
) -> c_int {
    if vocab_handle == 0 || token_ptr.is_null() || token_ids_out.is_null() || count_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let vocab = &*(vocab_handle as *const Vocabulary);
        let token_cstr = CStr::from_ptr(token_ptr);
        let token_str = match token_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return ERROR_INVALID_UTF8,
        };
        
        match vocab.token_ids(token_str.as_bytes()) {
            Some(token_ids) => {
                let ids_vec: Vec<u32> = token_ids.clone();
                *count_out = ids_vec.len();
                
                if ids_vec.is_empty() {
                    *token_ids_out = std::ptr::null_mut();
                } else {
                    // ✅ Use stable manual approach instead of into_raw_parts
                    let ptr = ids_vec.as_ptr() as *mut u32;
                    std::mem::forget(ids_vec); // Transfer ownership
                    *token_ids_out = ptr;
                }
                SUCCESS
            }
            None => {
                *count_out = 0;
                *token_ids_out = std::ptr::null_mut();
                SUCCESS
            }
        }
    }
}

//=============================================================================
// Index Functions
//=============================================================================

/// Create index from regex and vocabulary
#[no_mangle]
pub extern "C" fn outlines_create_index(
    regex_ptr: *const c_char,
    vocab_handle: VocabularyHandle,
    index_handle: *mut IndexHandle
) -> c_int {
    if regex_ptr.is_null() || vocab_handle == 0 || index_handle.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let regex_cstr = CStr::from_ptr(regex_ptr);
        let regex_str = match regex_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return ERROR_INVALID_UTF8,
        };
        
        let vocab = &*(vocab_handle as *const Vocabulary);
        
        match Index::new(regex_str, vocab) {
            Ok(index) => {
                let boxed_index = Box::new(index);
                *index_handle = Box::into_raw(boxed_index) as IndexHandle;
                SUCCESS
            }
            Err(_) => ERROR_INDEX_CREATION
        }
    }
}

/// Get initial state from index
#[no_mangle]
pub extern "C" fn outlines_index_initial_state(
    index_handle: IndexHandle,
    state_out: *mut u32
) -> c_int {
    if index_handle == 0 || state_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let index = &*(index_handle as *const Index);
        *state_out = index.initial_state();
        SUCCESS
    }
}

/// Get allowed tokens for a given state
#[no_mangle]
pub extern "C" fn outlines_index_allowed_tokens(
    index_handle: IndexHandle,
    state: u32,
    tokens_out: *mut *mut u32,
    count_out: *mut usize
) -> c_int {
    if index_handle == 0 || tokens_out.is_null() || count_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let index = &*(index_handle as *const Index);
        match index.allowed_tokens(&state) {
            Some(tokens) => {
                *count_out = tokens.len();
                
                if tokens.is_empty() {
                    *tokens_out = std::ptr::null_mut();
                } else {
                    // ✅ Use stable manual approach instead of into_raw_parts
                    let ptr = tokens.as_ptr() as *mut u32;
                    std::mem::forget(tokens); // Transfer ownership
                    *tokens_out = ptr;
                }
                SUCCESS
            }
            None => ERROR_INVALID_STATE
        }
    }
}

/// Get next state from current state and token
#[no_mangle]
pub extern "C" fn outlines_index_next_state(
    index_handle: IndexHandle,
    current_state: u32,
    token_id: u32,
    next_state_out: *mut u32
) -> c_int {
    if index_handle == 0 || next_state_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let index = &*(index_handle as *const Index);
        match index.next_state(&current_state, &token_id) {
            Some(next_state) => {
                *next_state_out = next_state;
                SUCCESS
            }
            None => ERROR_INVALID_STATE
        }
    }
}

/// Check if state is final
#[no_mangle]
pub extern "C" fn outlines_index_is_final_state(
    index_handle: IndexHandle,
    state: u32,
    is_final_out: *mut bool
) -> c_int {
    if index_handle == 0 || is_final_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let index = &*(index_handle as *const Index);
        *is_final_out = index.is_final_state(&state);
        SUCCESS
    }
}

/// Get all final states
#[no_mangle]
pub extern "C" fn outlines_index_final_states(
    index_handle: IndexHandle,
    states_out: *mut *mut u32,
    count_out: *mut usize
) -> c_int {
    if index_handle == 0 || states_out.is_null() || count_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        let index = &*(index_handle as *const Index);
        let final_states = index.final_states();
        let states_vec: Vec<u32> = final_states.iter().cloned().collect();
        
        *count_out = states_vec.len();
        
        if states_vec.is_empty() {
            *states_out = std::ptr::null_mut();
        } else {
            // ✅ Use stable manual approach instead of into_raw_parts
            let ptr = states_vec.as_ptr() as *mut u32;
            std::mem::forget(states_vec); // Transfer ownership
            *states_out = ptr;
        }
        SUCCESS
    }
}

//=============================================================================
// Guide Functions (High-level interface)
//=============================================================================

/// Create a guide from an index (similar to Python API)
#[no_mangle]
pub extern "C" fn outlines_create_guide(
    index_handle: IndexHandle,
    guide_handle: *mut GuideHandle
) -> c_int {
    if index_handle == 0 || guide_handle.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    unsafe {
        // For now, we'll use the index directly as a guide
        // In a full implementation, you'd create a wrapper struct
        *guide_handle = index_handle; // Simple mapping for now
        SUCCESS
    }
}

/// Get current state of the guide
#[no_mangle]
pub extern "C" fn outlines_guide_get_state(
    guide_handle: GuideHandle,
    state_out: *mut u32
) -> c_int {
    // For this simple implementation, we'll just return the initial state
    // In a full implementation, you'd track the current state in a Guide struct
    outlines_index_initial_state(guide_handle, state_out)
}

/// Get allowed tokens for current state of the guide
#[no_mangle]
pub extern "C" fn outlines_guide_get_tokens(
    guide_handle: GuideHandle,
    current_state: u32,
    tokens_out: *mut *mut u32,
    count_out: *mut usize
) -> c_int {
    outlines_index_allowed_tokens(guide_handle, current_state, tokens_out, count_out)
}

/// Advance guide to next state and return allowed tokens
#[no_mangle]
pub extern "C" fn outlines_guide_advance(
    guide_handle: GuideHandle,
    current_state: u32,
    token_id: u32,
    next_state_out: *mut u32,
    tokens_out: *mut *mut u32,
    count_out: *mut usize
) -> c_int {
    if guide_handle == 0 || next_state_out.is_null() || tokens_out.is_null() || count_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    // Get next state
    let result = outlines_index_next_state(guide_handle, current_state, token_id, next_state_out);
    if result != SUCCESS {
        return result;
    }
    
    // Get allowed tokens for next state
    unsafe {
        outlines_index_allowed_tokens(guide_handle, *next_state_out, tokens_out, count_out)
    }
}

/// Check if guide is finished (current state is final)
#[no_mangle]
pub extern "C" fn outlines_guide_is_finished(
    guide_handle: GuideHandle,
    current_state: u32,
    is_finished_out: *mut bool
) -> c_int {
    outlines_index_is_final_state(guide_handle, current_state, is_finished_out)
}

//=============================================================================
// Memory Management Functions
//=============================================================================

/// Free string allocated by Rust
#[no_mangle]
pub extern "C" fn outlines_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

/// Free token array allocated by Rust
#[no_mangle]
pub extern "C" fn outlines_free_tokens(tokens_ptr: *mut u32, count: usize) {
    if !tokens_ptr.is_null() && count > 0 {
        unsafe {
            let _ = Vec::from_raw_parts(tokens_ptr, count, count);
        }
    }
}

/// Free vocabulary handle
#[no_mangle]
pub extern "C" fn outlines_free_vocabulary(vocab_handle: VocabularyHandle) {
    if vocab_handle != 0 {
        unsafe {
            let _ = Box::from_raw(vocab_handle as *mut Vocabulary);
        }
    }
}

/// Free index handle
#[no_mangle]
pub extern "C" fn outlines_free_index(index_handle: IndexHandle) {
    if index_handle != 0 {
        unsafe {
            let _ = Box::from_raw(index_handle as *mut Index);
        }
    }
}

/// Free guide handle
#[no_mangle]
pub extern "C" fn outlines_free_guide(guide_handle: GuideHandle) {
    // For this simple implementation, guide and index are the same
    // In a full implementation, you'd have separate cleanup
    outlines_free_index(guide_handle);
}

//=============================================================================
// Utility Functions
//=============================================================================

/// Get version string
#[no_mangle]
pub extern "C" fn outlines_version(version_out: *mut *mut c_char) -> c_int {
    if version_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    let version = env!("CARGO_PKG_VERSION");
    let c_string = match CString::new(version) {
        Ok(cs) => cs,
        Err(_) => return ERROR_STRING_CONVERSION,
    };
    
    unsafe {
        *version_out = c_string.into_raw();
    }
    SUCCESS
}

/// Get error message for error code
#[no_mangle]
pub extern "C" fn outlines_error_message(error_code: c_int, message_out: *mut *mut c_char) -> c_int {
    if message_out.is_null() {
        return ERROR_NULL_POINTER;
    }
    
    let message = match error_code {
        SUCCESS => "Success",
        ERROR_NULL_POINTER => "Null pointer provided",
        ERROR_INVALID_UTF8 => "Invalid UTF-8 string",
        ERROR_STRING_CONVERSION => "String conversion error",
        ERROR_SCHEMA_PARSING => "JSON schema parsing error",
        ERROR_VOCABULARY_CREATION => "Vocabulary creation error",
        ERROR_INDEX_CREATION => "Index creation error",
        ERROR_REGEX_COMPILATION => "Regex compilation error",
        ERROR_INVALID_STATE => "Invalid state",
        ERROR_INVALID_TOKEN => "Invalid token",
        ERROR_GUIDE_CREATION => "Guide creation error",
        _ => "Unknown error",
    };
    
    let c_string = match CString::new(message) {
        Ok(cs) => cs,
        Err(_) => return ERROR_STRING_CONVERSION,
    };
    
    unsafe {
        *message_out = c_string.into_raw();
    }
    SUCCESS
}