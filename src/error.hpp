#pragma once
#include "sam3d.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace s3d {
inline void message(char *error, uint64_t capacity, const char *text) noexcept {
    if (!error || !capacity) return;
    const auto n = std::min<uint64_t>(capacity - 1, std::strlen(text));
    std::memcpy(error, text, static_cast<size_t>(n)); error[n] = '\0';
}
template<class F> s3d_status boundary(char *error, uint64_t capacity, F &&body) noexcept {
    if ((!error && capacity) || capacity > std::numeric_limits<size_t>::max()) return S3D_INVALID_ARGUMENT;
    try { body(); message(error, capacity, ""); return S3D_OK; }
    catch (const std::invalid_argument &e) { message(error, capacity, e.what()); return S3D_INVALID_ARGUMENT; }
    catch (const std::bad_alloc &) { message(error, capacity, "allocation failed"); return S3D_OUT_OF_MEMORY; }
    catch (const std::exception &e) { message(error, capacity, e.what()); return S3D_INTERNAL_ERROR; }
    catch (...) { message(error, capacity, "unknown internal error"); return S3D_INTERNAL_ERROR; }
}
inline void require(bool condition, const char *text) {
    if (!condition) throw std::invalid_argument(text);
}
}
