// C ABI over Qwen3ASR / ForcedAligner, see include/qwen3_asr_c.h.

#include "qwen3_asr_c.h"

#include "forced_aligner.h"
#include "qwen3_asr.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

void set_error(std::string msg) { g_last_error = std::move(msg); }

int threads_or_default(int n_threads) { return n_threads > 0 ? n_threads : 4; }

} // namespace

struct qwen3_asr_context {
    qwen3_asr::Qwen3ASR asr;
    int n_threads = 4;
};

struct qwen3_aligner_context {
    qwen3_asr::ForcedAligner aligner;
};

extern "C" {

const char * qwen3_asr_last_error(void) { return g_last_error.c_str(); }

struct qwen3_asr_context * qwen3_asr_init(const char * model_path, int n_threads) {
    if (!model_path) {
        set_error("qwen3_asr_init: model_path is NULL");
        return nullptr;
    }
    try {
        auto * ctx = new qwen3_asr_context();
        ctx->n_threads = threads_or_default(n_threads);
        if (!ctx->asr.load_model(model_path)) {
            set_error(ctx->asr.get_error());
            delete ctx;
            return nullptr;
        }
        return ctx;
    } catch (const std::exception & e) {
        set_error(std::string("qwen3_asr_init: ") + e.what());
        return nullptr;
    }
}

void qwen3_asr_free(struct qwen3_asr_context * ctx) { delete ctx; }

char * qwen3_asr_transcribe(struct qwen3_asr_context * ctx, const float * pcm16k, int n_samples,
                            int max_tokens) {
    if (!ctx || !pcm16k || n_samples <= 0) {
        set_error("qwen3_asr_transcribe: invalid arguments");
        return nullptr;
    }
    try {
        qwen3_asr::transcribe_params params;
        params.n_threads = ctx->n_threads;
        params.max_tokens = max_tokens > 0 ? max_tokens : 1024;
        params.print_progress = false;
        params.print_timing = false;
        qwen3_asr::transcribe_result r = ctx->asr.transcribe(pcm16k, n_samples, params);
        if (!r.success) {
            set_error(r.error_msg);
            return nullptr;
        }
        char * out = static_cast<char *>(std::malloc(r.text.size() + 1));
        if (!out) {
            set_error("qwen3_asr_transcribe: out of memory");
            return nullptr;
        }
        std::memcpy(out, r.text.c_str(), r.text.size() + 1);
        return out;
    } catch (const std::exception & e) {
        set_error(std::string("qwen3_asr_transcribe: ") + e.what());
        return nullptr;
    }
}

void qwen3_asr_string_free(char * s) { std::free(s); }

struct qwen3_aligner_context * qwen3_aligner_init(const char * model_path, int n_threads) {
    if (!model_path) {
        set_error("qwen3_aligner_init: model_path is NULL");
        return nullptr;
    }
    try {
        auto * ctx = new qwen3_aligner_context();
        if (!ctx->aligner.load_model(model_path)) {
            set_error(ctx->aligner.get_error());
            delete ctx;
            return nullptr;
        }
        ctx->aligner.set_n_threads(threads_or_default(n_threads));
        return ctx;
    } catch (const std::exception & e) {
        set_error(std::string("qwen3_aligner_init: ") + e.what());
        return nullptr;
    }
}

void qwen3_aligner_free(struct qwen3_aligner_context * ctx) { delete ctx; }

int qwen3_aligner_align(struct qwen3_aligner_context * ctx, const float * pcm16k, int n_samples,
                        const char * const * words, int n_words, const char * language,
                        struct qwen3_word_span * out_spans) {
    (void) language;
    if (!ctx || !pcm16k || n_samples <= 0 || !words || n_words <= 0 || !out_spans) {
        set_error("qwen3_aligner_align: invalid arguments");
        return -1;
    }
    try {
        std::vector<std::string> list;
        list.reserve(static_cast<size_t>(n_words));
        for (int i = 0; i < n_words; ++i) {
            if (!words[i]) {
                set_error("qwen3_aligner_align: words[" + std::to_string(i) + "] is NULL");
                return -1;
            }
            list.emplace_back(words[i]);
        }
        qwen3_asr::alignment_result r = ctx->aligner.align_words(pcm16k, n_samples, list);
        if (!r.success) {
            set_error(r.error_msg);
            return -2;
        }
        if (r.words.size() != list.size()) {
            set_error("qwen3_aligner_align: got " + std::to_string(r.words.size()) +
                      " spans for " + std::to_string(list.size()) + " words");
            return -2;
        }
        for (size_t i = 0; i < r.words.size(); ++i) {
            out_spans[i].start_s = r.words[i].start;
            out_spans[i].end_s = r.words[i].end;
        }
        return 0;
    } catch (const std::exception & e) {
        set_error(std::string("qwen3_aligner_align: ") + e.what());
        return -2;
    }
}

} // extern "C"
