// tools/v41/engine/ds41_io.hpp - the file and memory calls of the DeepSeek-V4.1 engine, as `dio::`.
//
// The basic calls (open / pread / pwrite / fstat / fsync / posix_fadvise / setenv) are ds41_sys.hpp's: thin POSIX
// wrappers on Linux, overlapped Win32 handles on Windows (O_DIRECT = FILE_FLAG_NO_BUFFERING).  This header adds what
// the Windows tuning needs on top: the volume's sector size for in-place unbuffered reads (direct_align), free disk
// space (the blob mirror's builder), MemAvailable in GiB, and a heap check for debugging.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "ds41_sys.hpp"

#if !defined(_WIN32)
#include <strings.h>
#include <sys/statvfs.h>
#endif

namespace strata::v41::dio {

using ds41_sys::Stat;
using ds41_sys::open;
using ds41_sys::close;
using ds41_sys::pread;
using ds41_sys::pwrite;
using ds41_sys::fstat;
using ds41_sys::fsync;
using ds41_sys::posix_fadvise;
using ds41_sys::setenv;

/// MemAvailable in GiB (Windows: ullAvailPhys), -1 when unknown.
inline int64_t mem_available_gib() {
    const uint64_t b = ds41_sys::mem_available_bytes();
    return b == 0 ? -1 : (int64_t) (b >> 30);
}

#if defined(_WIN32)

using ds41_sys::widen;

inline bool exists(const std::string& p) { return GetFileAttributesW(widen(p.c_str()).c_str()) != INVALID_FILE_ATTRIBUTES; }

/// The alignment a FILE_FLAG_NO_BUFFERING transfer on `fd` needs for its offset, length and buffer: the volume's
/// logical sector size (512 on the 990 PRO here, so the expert spans that sit 2 KiB off a page can still be read
/// straight into their destination).  0 when unknown.
inline size_t direct_align(int fd) {
    FILE_STORAGE_INFO si{};
    if (!GetFileInformationByHandleEx(ds41_sys::handle_of(fd), FileStorageInfo, &si, sizeof si)) return 0;
    const size_t a = si.LogicalBytesPerSector;
    return a >= 512 && (a & (a - 1)) == 0 ? a : 0;
}

inline void* aligned_alloc(size_t align, size_t n) { return _aligned_malloc(n ? n : align, align); }
inline void aligned_free(void* p) { _aligned_free(p); }

inline int strcasecmp(const char* a, const char* b) { return _stricmp(a, b); }

/// Free and total bytes of the volume holding `dir` (a directory that exists).  False when unknown.
inline bool disk_space(const std::string& dir, uint64_t& free_b, uint64_t& total_b) {
    ULARGE_INTEGER fr{}, tot{};
    if (!GetDiskFreeSpaceExW(ds41_sys::widen(dir.c_str()).c_str(), &fr, &tot, nullptr)) return false;
    free_b = fr.QuadPart;
    total_b = tot.QuadPart;
    return true;
}

#else  // POSIX

inline bool exists(const std::string& p) { struct ::stat st; return ::stat(p.c_str(), &st) == 0; }
inline size_t direct_align(int) { return 0; }   // Linux keeps its bounce path (read_inplace off)
// posix_memalign, not std::aligned_alloc: callers pass sizes that are not a multiple of the alignment
inline void* aligned_alloc(size_t align, size_t n) { void* p = nullptr; return ::posix_memalign(&p, align, n) == 0 ? p : nullptr; }
inline void aligned_free(void* p) { std::free(p); }
inline int strcasecmp(const char* a, const char* b) { return ::strcasecmp(a, b); }
inline bool disk_space(const std::string& dir, uint64_t& free_b, uint64_t& total_b) {
    struct ::statvfs v;
    if (::statvfs(dir.c_str(), &v) != 0) return false;
    free_b = (uint64_t) v.f_bavail * (uint64_t) v.f_frsize;
    total_b = (uint64_t) v.f_blocks * (uint64_t) v.f_frsize;
    return true;
}
#endif

/// DS41_HEAP_CHECK=1 (Windows, debugging): validate every heap of the process (the CRT's and the drivers' own) and
/// say which one is damaged, so a heap overwrite can be pinned to the step that made it.  A no-op elsewhere.
inline void heap_check(const char* where) {
#if defined(_WIN32)
    static const bool on = std::getenv("DS41_HEAP_CHECK") != nullptr;
    if (!on) return;
    HANDLE hs[256];
    const DWORD n = GetProcessHeaps(256, hs);
    const HANDLE ph = GetProcessHeap();
    int bad = 0;
    for (DWORD i = 0; i < n && i < 256; ++i)
        if (!HeapValidate(hs[i], 0, nullptr)) {
            ++bad;
            std::fprintf(stderr, "heap check %s: heap %lu of %lu (%p%s) CORRUPT\n", where, i, n, hs[i],
                         hs[i] == ph ? ", the process heap" : "");
        }
    if (!bad) std::fprintf(stderr, "heap check %s: %lu heaps ok\n", where, n);
    std::fflush(stderr);
#else
    (void) where;
#endif
}

}  // namespace strata::v41::dio
