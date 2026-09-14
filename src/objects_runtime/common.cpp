#include "common.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace sam3d {

static LogLevel g_log_level = LogLevel::Info;
static std::mutex g_log_mutex;

namespace {
struct LogLevelInit {
    LogLevelInit() {
        const char* env = getenv("SAM3D_LOG_LEVEL");
        if (!env) return;
        if (!strcmp(env, "debug") || !strcmp(env, "0")) g_log_level = LogLevel::Debug;
        else if (!strcmp(env, "warn")) g_log_level = LogLevel::Warn;
        else if (!strcmp(env, "error")) g_log_level = LogLevel::Error;
    }
};
LogLevelInit g_log_level_init;
}  // namespace

void set_log_level(LogLevel level) { g_log_level = level; }
LogLevel log_level() { return g_log_level; }

void log_printf(LogLevel level, const char* fmt, ...) {
    if (level < g_log_level) return;
    const char* tag = "I";
    switch (level) {
        case LogLevel::Debug: tag = "D"; break;
        case LogLevel::Info: tag = "I"; break;
        case LogLevel::Warn: tag = "W"; break;
        case LogLevel::Error: tag = "E"; break;
    }
    std::lock_guard<std::mutex> lock(g_log_mutex);
    fprintf(stderr, "[%s] ", tag);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    fflush(stderr);
}

double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

ScopedTimer::ScopedTimer(const char* name_) : name(name_), t0(now_ms()) {}
ScopedTimer::~ScopedTimer() {
    LOGD("%s took %.2f ms", name, now_ms() - t0);
}

static constexpr uint32_t kSamtMagic = 0x544D4153;  // "SAMT" little endian

bool save_raw_tensor(const std::string& path, const RawTensor& t) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const int32_t nd = (int32_t)t.ne.size();
    const int32_t type = (int32_t)t.type;
    bool ok = fwrite(&kSamtMagic, 4, 1, f) == 1 && fwrite(&nd, 4, 1, f) == 1;
    ok = ok && fwrite(t.ne.data(), 8, nd, f) == (size_t)nd;
    ok = ok && fwrite(&type, 4, 1, f) == 1;
    const size_t nbytes = t.data.size();
    ok = ok && fwrite(t.data.data(), 1, nbytes, f) == nbytes;
    fclose(f);
    return ok;
}

bool save_raw_tensor_f32(const std::string& path, const std::vector<int64_t>& ne,
                         const float* data) {
    RawTensor t;
    t.ne = ne;
    t.type = GGML_TYPE_F32;
    size_t n = 1;
    for (int64_t d : ne) n *= (size_t)d;
    t.data.resize(n * sizeof(float));
    memcpy(t.data.data(), data, n * sizeof(float));
    return save_raw_tensor(path, t);
}

bool load_raw_tensor(const std::string& path, RawTensor& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    int32_t nd = 0, type = 0;
    bool ok = fread(&magic, 4, 1, f) == 1 && magic == kSamtMagic &&
              fread(&nd, 4, 1, f) == 1 && nd > 0 && nd <= 8;
    if (!ok) {
        fclose(f);
        return false;
    }
    out.ne.resize(nd);
    ok = fread(out.ne.data(), 8, nd, f) == (size_t)nd &&
         fread(&type, 4, 1, f) == 1;
    if (!ok) {
        fclose(f);
        return false;
    }
    // SAMT writers have used both 26 (this ggml's GGML_TYPE_I32) and 30
    // (upstream's newer value) for int32 tensors; normalize to this build.
    out.type = (enum ggml_type)type;
    if (out.type == (enum ggml_type)30) out.type = GGML_TYPE_I32;
    size_t n = 1;
    for (int64_t d : out.ne) n *= (size_t)d;
    const size_t nbytes = n * ggml_type_size(out.type) / ggml_blck_size(out.type);
    out.data.resize(nbytes);
    ok = fread(out.data.data(), 1, nbytes, f) == nbytes;
    fclose(f);
    return ok;
}

void copy_tensor_to_float(const struct ggml_tensor* t, std::vector<float>& out) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const size_t n = ggml_nelements(t);
    const size_t base = out.size();
    out.resize(base + n);
    // ggml tensors may be non-contiguous when viewed; go through strides.
    if (ggml_is_contiguous(t)) {
        memcpy(out.data() + base, t->data, n * sizeof(float));
        return;
    }
    // fallback: element-wise (small tensors only in practice)
    std::vector<float> tmp(n);
    const int nd = ggml_n_dims(t);
    std::vector<int64_t> idx(nd, 0);
    for (size_t i = 0; i < n; i++) {
        size_t offset = 0;
        size_t stride = 1;
        for (int d = 0; d < nd; d++) {
            offset += (size_t)idx[d] * (size_t)t->nb[d];
        }
        (void)stride;
        tmp[i] = *(const float*)((const char*)t->data + offset);
        for (int d = 0; d < nd; d++) {
            idx[d]++;
            if (idx[d] < t->ne[d]) break;
            idx[d] = 0;
        }
    }
    memcpy(out.data() + base, tmp.data(), n * sizeof(float));
}

bool read_file_bytes(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(size);
    const bool ok = fread(out.data(), 1, size, f) == (size_t)size;
    fclose(f);
    return ok;
}

bool write_file_bytes(const std::string& path, const void* data, size_t nbytes) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = fwrite(data, 1, nbytes, f) == nbytes;
    fclose(f);
    return ok;
}

bool file_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default: out += c;
        }
    }
    return out;
}

}  // namespace sam3d
