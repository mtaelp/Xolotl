// tools/v41/engine/ds41_io.hpp - the POSIX file calls the DeepSeek-V4.1 engine makes, on Linux and on Windows.
//
// Linux: thin inline wrappers over open / pread / pwrite / fstat / posix_fadvise / posix_memalign, so the Linux
// build calls exactly what it called before.
// Windows: the same calls on Win32 handles.  Every file is opened FILE_FLAG_OVERLAPPED, so the engine's reader
// threads (32 by default) issue positioned reads in parallel on one handle - a synchronous handle would serialise
// them in the I/O manager.  O_DIRECT maps to FILE_FLAG_NO_BUFFERING (the callers already align offset, length and
// buffer to 4 KiB, which is what that flag needs).  The handle is wrapped in a CRT descriptor (_open_osfhandle), so
// the callers keep their `int fd`.  posix_fadvise has no Windows counterpart and does nothing there.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <malloc.h>
#include <cstring>
#include <type_traits>
using ssize_t = std::make_signed_t<size_t>;
#ifndef O_DIRECT
#define O_DIRECT 0x40000000
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef POSIX_FADV_NORMAL
#define POSIX_FADV_NORMAL 0
#define POSIX_FADV_RANDOM 1
#define POSIX_FADV_SEQUENTIAL 2
#define POSIX_FADV_WILLNEED 3
#define POSIX_FADV_DONTNEED 4
#define POSIX_FADV_NOREUSE 5
#endif
#else
#include <fcntl.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace strata::v41::dio {

#if defined(_WIN32)

struct Stat {
    uint64_t st_size = 0;
    uint64_t st_ino = 0;   ///< the file index: equal for two handles of one file
    uint64_t st_dev = 0;   ///< the volume serial number
};

inline std::wstring widen(const char* s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring w(n > 0 ? (size_t) n : 0, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    if (!w.empty() && w.back() == L'\0') w.pop_back();
    return w;
}

inline HANDLE handle(int fd) { return (HANDLE) _get_osfhandle(fd); }

inline int open(const char* path, int flags, int /*mode*/ = 0) {
    const int acc = flags & (O_RDONLY | O_WRONLY | O_RDWR);
    const DWORD access = acc == O_WRONLY ? GENERIC_WRITE : acc == O_RDWR ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    DWORD disp = OPEN_EXISTING;
    if (flags & O_CREAT) disp = (flags & O_TRUNC) ? CREATE_ALWAYS : OPEN_ALWAYS;
    else if (flags & O_TRUNC) disp = TRUNCATE_EXISTING;
    DWORD attrs = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
    if (flags & O_DIRECT) attrs |= FILE_FLAG_NO_BUFFERING;
    const HANDLE h = CreateFileW(widen(path).c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                 nullptr, disp, attrs, nullptr);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const int fd = _open_osfhandle((intptr_t) h, acc == O_RDONLY ? _O_RDONLY : 0);
    if (fd < 0) CloseHandle(h);
    return fd;
}

inline int close(int fd) { return _close(fd); }   // closes the Win32 handle too

/// One positioned transfer on an overlapped handle, waited for on this thread's own event.
inline ssize_t xfer(int fd, void* buf, size_t n, int64_t off, bool write) {
    const HANDLE h = handle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    struct Ev { HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr); ~Ev() { if (e) CloseHandle(e); } };
    thread_local Ev ev;
    const DWORD want = (DWORD) (n > (1u << 30) ? (1u << 30) : n);   // short transfers are allowed: callers loop
    OVERLAPPED ov{};
    ov.Offset = (DWORD) ((uint64_t) off & 0xffffffffu);
    ov.OffsetHigh = (DWORD) ((uint64_t) off >> 32);
    ov.hEvent = ev.e;
    ResetEvent(ev.e);
    DWORD got = 0;
    const BOOL ok = write ? WriteFile(h, buf, want, nullptr, &ov) : ReadFile(h, buf, want, nullptr, &ov);
    if (!ok) {
        const DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) return 0;
        if (e != ERROR_IO_PENDING) return -1;
    }
    if (!GetOverlappedResult(h, &ov, &got, TRUE)) return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    return (ssize_t) got;
}

inline ssize_t pread(int fd, void* buf, size_t n, int64_t off) { return xfer(fd, buf, n, off, false); }
inline ssize_t pwrite(int fd, const void* buf, size_t n, int64_t off) { return xfer(fd, (void*) buf, n, off, true); }

inline int fstat(int fd, Stat* st) {
    BY_HANDLE_FILE_INFORMATION fi;
    if (!GetFileInformationByHandle(handle(fd), &fi)) return -1;
    st->st_size = ((uint64_t) fi.nFileSizeHigh << 32) | fi.nFileSizeLow;
    st->st_ino = ((uint64_t) fi.nFileIndexHigh << 32) | fi.nFileIndexLow;
    st->st_dev = fi.dwVolumeSerialNumber;
    return 0;
}

inline bool exists(const std::string& p) { return GetFileAttributesW(widen(p.c_str()).c_str()) != INVALID_FILE_ATTRIBUTES; }

inline int fsync(int fd) { return FlushFileBuffers(handle(fd)) ? 0 : -1; }

inline int posix_fadvise(int, int64_t, int64_t, int) { return 0; }

/// The alignment a FILE_FLAG_NO_BUFFERING transfer on `fd` needs for its offset, length and buffer: the volume's
/// logical sector size (512 on the 990 PRO here, so the expert spans that sit 2 KiB off a page can still be read
/// straight into their destination).  0 when unknown.
inline size_t direct_align(int fd) {
    FILE_STORAGE_INFO si{};
    if (!GetFileInformationByHandleEx(handle(fd), FileStorageInfo, &si, sizeof si)) return 0;
    const size_t a = si.LogicalBytesPerSector;
    return a >= 512 && (a & (a - 1)) == 0 ? a : 0;
}

inline void* aligned_alloc(size_t align, size_t n) { return _aligned_malloc(n ? n : align, align); }
inline void aligned_free(void* p) { _aligned_free(p); }

inline int setenv(const char* k, const char* v, int overwrite) {
    if (!overwrite && std::getenv(k)) return 0;
    return _putenv_s(k, v);
}

inline int strcasecmp(const char* a, const char* b) { return _stricmp(a, b); }

/// MemAvailable in GiB (Linux: /proc/meminfo's MemAvailable), -1 when unknown.
inline int64_t mem_available_gib() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    if (!GlobalMemoryStatusEx(&m)) return -1;
    return (int64_t) (m.ullAvailPhys >> 30);
}

/// Free and total bytes of the volume holding `dir` (a directory that exists).  False when unknown.
inline bool disk_space(const std::string& dir, uint64_t& free_b, uint64_t& total_b) {
    ULARGE_INTEGER fr{}, tot{};
    if (!GetDiskFreeSpaceExW(widen(dir.c_str()).c_str(), &fr, &tot, nullptr)) return false;
    free_b = fr.QuadPart;
    total_b = tot.QuadPart;
    return true;
}

#else  // POSIX

using Stat = struct ::stat;
inline int open(const char* path, int flags, int mode = 0) { return ::open(path, flags, mode); }
inline int close(int fd) { return ::close(fd); }
inline ssize_t pread(int fd, void* buf, size_t n, int64_t off) { return ::pread(fd, buf, n, (off_t) off); }
inline ssize_t pwrite(int fd, const void* buf, size_t n, int64_t off) { return ::pwrite(fd, buf, n, (off_t) off); }
inline int fstat(int fd, Stat* st) { return ::fstat(fd, st); }
inline bool exists(const std::string& p) { struct ::stat st; return ::stat(p.c_str(), &st) == 0; }
inline int fsync(int fd) { return ::fsync(fd); }
inline int posix_fadvise(int fd, int64_t off, int64_t len, int adv) { return ::posix_fadvise(fd, (off_t) off, (off_t) len, adv); }
inline size_t direct_align(int) { return 0; }   // Linux keeps its bounce path (read_inplace off)
inline void* aligned_alloc(size_t align, size_t n) { void* p = nullptr; return ::posix_memalign(&p, align, n) == 0 ? p : nullptr; }
inline void aligned_free(void* p) { std::free(p); }
inline int setenv(const char* k, const char* v, int overwrite) { return ::setenv(k, v, overwrite); }
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
