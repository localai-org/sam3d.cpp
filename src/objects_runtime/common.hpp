// Common helpers: logging, timing, raw tensor IO for parity checks.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

namespace sam3d {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

void set_log_level(LogLevel level);
LogLevel log_level();
void log_printf(LogLevel level, const char* fmt, ...);

#define LOGD(...) ::sam3d::log_printf(::sam3d::LogLevel::Debug, __VA_ARGS__)
#define LOGI(...) ::sam3d::log_printf(::sam3d::LogLevel::Info, __VA_ARGS__)
#define LOGW(...) ::sam3d::log_printf(::sam3d::LogLevel::Warn, __VA_ARGS__)
#define LOGE(...) ::sam3d::log_printf(::sam3d::LogLevel::Error, __VA_ARGS__)

// monotonic wall clock in milliseconds
double now_ms();

// Simple scoped stopwatch reporting to stderr at Debug level.
struct ScopedTimer {
    const char* name;
    double t0;
    explicit ScopedTimer(const char* name_);
    ~ScopedTimer();
};

// ---- raw float tensor IO ---------------------------------------------------
// Binary format used by parity tooling and intermediate dumps (the ne[]
// entries are int64, matching ggml's int64_t* ne):
//   magic "SAMT" | int32 ndims | int64 ne[ndims] | int32 type | data
struct RawTensor {
    std::vector<int64_t> ne;
    enum ggml_type type = GGML_TYPE_F32;
    std::vector<uint8_t> data;
};

bool save_raw_tensor(const std::string& path, const RawTensor& t);
bool save_raw_tensor_f32(const std::string& path, const std::vector<int64_t>& ne,
                         const float* data);
bool load_raw_tensor(const std::string& path, RawTensor& out);

// helper to append a float element/row to a vector
void copy_tensor_to_float(const struct ggml_tensor* t, std::vector<float>& out);

// Reads an entire file into memory; returns false on failure.
bool read_file_bytes(const std::string& path, std::vector<uint8_t>& out);
bool write_file_bytes(const std::string& path, const void* data, size_t nbytes);
bool file_exists(const std::string& path);

// Minimal JSON escaping for metadata output.
std::string json_escape(const std::string& s);

}  // namespace sam3d
