#pragma once
// qwen3_asr_c.h: C ABI over the Qwen3ASR and ForcedAligner C++ classes.
//
// Pure C99, for bindgen / ctypes / cgo consumers. Audio is always mono
// float32 PCM at 16 kHz, normalised to [-1, 1]. Handles are not thread
// safe: use one handle per thread (or serialise calls). Failing entries
// leave a message in qwen3_asr_last_error() (thread local, valid until
// the next failing call on the same thread).

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(QWEN3ASR_EXPORTS)
#    define QWEN3_ASR_C_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#    define QWEN3_ASR_C_API __attribute__((visibility("default")))
#else
#    define QWEN3_ASR_C_API
#endif

// Last error message on the calling thread, "" if none. Never NULL.
QWEN3_ASR_C_API const char * qwen3_asr_last_error(void);

// ---- transcription (Qwen3-ASR) ----------------------------------------

struct qwen3_asr_context;

// Load a Qwen3-ASR GGUF. n_threads <= 0 selects 4. NULL on failure.
QWEN3_ASR_C_API struct qwen3_asr_context * qwen3_asr_init(const char * model_path, int n_threads);

// Safe on NULL.
QWEN3_ASR_C_API void qwen3_asr_free(struct qwen3_asr_context * ctx);

// Transcribe pcm16k[0..n_samples). max_tokens <= 0 selects 1024. Returns
// a malloc allocated NUL terminated UTF-8 string (release it with
// qwen3_asr_string_free), or NULL on failure.
QWEN3_ASR_C_API char * qwen3_asr_transcribe(struct qwen3_asr_context * ctx,
                                            const float * pcm16k, int n_samples,
                                            int max_tokens);

// Safe on NULL.
QWEN3_ASR_C_API void qwen3_asr_string_free(char * s);

// ---- forced alignment (Qwen3-ForcedAligner) -----------------------------

struct qwen3_aligner_context;

// Start and end of one aligned word, in seconds from the start of the
// audio, clamped to the audio duration.
struct qwen3_word_span {
    float start_s;
    float end_s;
};

// Load a Qwen3-ForcedAligner GGUF. n_threads <= 0 selects 4. NULL on
// failure.
QWEN3_ASR_C_API struct qwen3_aligner_context * qwen3_aligner_init(const char * model_path, int n_threads);

// Safe on NULL.
QWEN3_ASR_C_API void qwen3_aligner_free(struct qwen3_aligner_context * ctx);

// Align words[0..n_words) (NUL terminated UTF-8, already split by the
// caller; no further splitting happens) to pcm16k[0..n_samples) and write
// one span per word, in order, to out_spans[0..n_words) (caller
// allocated). language is accepted for interface stability and currently
// unused: the aligner only uses the language to split a text into words,
// and here the caller supplies the words. Returns 0 on success, -1 on
// invalid arguments, -2 on alignment failure.
QWEN3_ASR_C_API int qwen3_aligner_align(struct qwen3_aligner_context * ctx,
                                        const float * pcm16k, int n_samples,
                                        const char * const * words, int n_words,
                                        const char * language,
                                        struct qwen3_word_span * out_spans);

#ifdef __cplusplus
}
#endif
