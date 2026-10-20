// ds41_rdbench.cpp - the engine's file-tier read pattern on its own (Windows): what the SSD gives, and what slows it.
//
// Reads the blobs listed by ds41_rdbench_spans.py (3 spans each: gate, up, down - at the model file's real offsets,
// layer by layer, ascending expert) with N threads pulling work off a shared counter, through dio::pread.
//
//   ds41_rdbench <model.gguf> <spans.bin> <direct|buf> <pread|split|parts> <threads> <chunk KiB | parts> <n|s|r>
//                <first> <count> [copy] [h2d GiB] [reg GiB] [warm GiB] [warm at GiB] [warm mode] [wait s]
//     pread: one dio::pread per span, widened to 4 KiB, into a bounce (Ds41FileBlobs::read_direct)
//     split: each blob's 3 spans cut into <chunk KiB> requests, all in flight at once, then waited for
//     parts: the prompt's wave reader - each blob cut into <parts> parts (Ds41FileBlobs::read_part)
//     n|s|r: no extra flag, FILE_FLAG_SEQUENTIAL_SCAN, FILE_FLAG_RANDOM_ACCESS
//     first, count: the blob range to read
//     copy: 1 (default) memcpy out of the bounce, 0 skip it, 2 (parts) read 512-aligned pieces straight into the
//           destination (Ds41FileBlobs::read_inplace)
//   Built with -DRDBENCH_CUDA (nvcc), the rest model the engine around the reads:
//     h2d: GiB of pinned memory copied to the card in a loop while reading (the prompt's arena DMA)
//     reg: GiB cudaHostRegister'd first (the 64 GiB RAM arena); with h2d, the copies come from it in 10 MiB pieces
//     warm: GiB read first through a second, buffered handle from <warm at> GiB; warm mode: 0 keep the handle open,
//           1 close it, o open it only, m map the file and touch the range through the view (kept), M map, touch,
//           unmap and close; wait: seconds before the reads
//   Env: RDB_PRE_BUF=1 opens a buffered handle before the unbuffered one; RDB_STORINFO=1 queries dio::direct_align;
//        RDB_GGUF=1 maps and unmaps the file as GgufModel::open does.
//
// Build (x64 Native Tools prompt): cl /O2 /EHsc /std:c++17 /I tools41\engine tools41\engine\ds41_rdbench.cpp
//   CUDA: nvcc -O2 -std=c++17 -DRDBENCH_CUDA -x cu -I tools41\engine tools41\engine\ds41_rdbench.cpp
#include "ds41_io.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#if defined(RDBENCH_CUDA)
#include <cuda_runtime.h>
#endif

namespace dio = strata::v41::dio;

struct Span { uint64_t off, bytes; };

int main(int argc, char** argv) {
#if !defined(_WIN32)
    std::fprintf(stderr, "ds41_rdbench: Windows only\n");
    return 1;
#else
    if (argc < 10) {
        std::fprintf(stderr, "usage: ds41_rdbench <model.gguf> <spans.bin> <direct|buf> <pread|split> <threads> "
                             "<chunk KiB> <n|s|r> <first> <count> [copy]\n");
        return 2;
    }
    const bool direct = std::strcmp(argv[3], "direct") == 0;
    const bool split = std::strcmp(argv[4], "split") == 0;
    const bool parts = std::strcmp(argv[4], "parts") == 0;
    const int nth = std::max(1, std::atoi(argv[5]));
    const uint64_t chunk = (uint64_t) std::max(4, std::atoi(argv[6])) * 1024;
    const int P = std::max(1, std::atoi(argv[6]));   // parts: the part count
    const char fl = argv[7][0];
    const int64_t first = std::atoll(argv[8]);
    int64_t count = std::atoll(argv[9]);
    const bool copy = argc < 11 || argv[10][0] != '0';
    const bool inplace = argc >= 11 && argv[10][0] == '2';   // parts: read 512-aligned pieces straight into dst

    std::vector<Span> sp;
    {
        FILE* f = std::fopen(argv[2], "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        Span s;
        while (std::fread(&s, sizeof s, 1, f) == 1) sp.push_back(s);
        std::fclose(f);
    }
    const int64_t nblobs = (int64_t) sp.size() / 3;
    if (first >= nblobs) { std::fprintf(stderr, "first >= %lld blobs\n", (long long) nblobs); return 1; }
    count = std::min(count, nblobs - first);

    DWORD attrs = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
    if (direct) attrs |= FILE_FLAG_NO_BUFFERING;
    if (fl == 's') attrs |= FILE_FLAG_SEQUENTIAL_SCAN;
    if (fl == 'r') attrs |= FILE_FLAG_RANDOM_ACCESS;
    // RDB_PRE_BUF=1: open a BUFFERED handle to the file first (kept open, never read) - Ds41FileBlobs::open's order
    if (std::getenv("RDB_PRE_BUF")) std::printf("  pre-opened buffered fd %d%c", dio::open(argv[1], O_RDONLY), 10);
    const HANDLE h = CreateFileW(dio::widen(argv[1]).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                 nullptr, OPEN_EXISTING, attrs, nullptr);
    if (h == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "CreateFile failed %lu\n", GetLastError()); return 1; }
    const int fd = _open_osfhandle((intptr_t) h, _O_RDONLY);
    // RDB_STORINFO=1: query the volume's sector sizes on the handle (dio::direct_align)
    if (std::getenv("RDB_STORINFO")) std::printf("  direct_align %zu%c", dio::direct_align(fd), 10);
    // RDB_GGUF=1: what GgufModel::open does - CreateFileA (share read), map the whole file, read the header pages, unmap
    if (std::getenv("RDB_GGUF")) {
        HANDLE gh = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        HANDLE gm = CreateFileMappingA(gh, nullptr, PAGE_READONLY, 0, 0, nullptr);
        const volatile uint8_t* gv = (const volatile uint8_t*) MapViewOfFile(gm, FILE_MAP_READ, 0, 0, 0);
        uint64_t sum = 0;
        for (uint64_t q = 0; q < (16u << 20); q += 4096) sum += gv[q];
        UnmapViewOfFile((const void*) gv);
        CloseHandle(gm);
        CloseHandle(gh);
        std::printf("  gguf-style map/unmap done (%llu)%c", (unsigned long long) (sum & 1), 10);
    }

    uint64_t maxspan = 0;
    for (const Span& s : sp) maxspan = std::max(maxspan, s.bytes);
    const size_t bounce = (size_t) ((maxspan + 8192 + 4095) & ~4095ull);

    std::atomic<bool> h2d_stop{false};
    std::atomic<int64_t> h2d_bytes{0};
    std::thread h2d;
#if defined(RDBENCH_CUDA)
    const double h2d_gib = argc > 11 ? std::atof(argv[11]) : 0.0;
    // reg (argv 12): GiB of ordinary memory cudaHostRegister'd before the reads (the engine pins its 64 GiB RAM arena)
    const double reg_gib = argc > 12 ? std::atof(argv[12]) : 0.0;
    // warm (argv 13): GiB of the file read first through a second, BUFFERED handle that stays open (the engine seeds its
    // RAM arena through its buffered fd, so the file has a cache map while the prompt's unbuffered reads run)
    HANDLE wh = INVALID_HANDLE_VALUE;
    if (argc > 13 && std::atof(argv[13]) > 0) {
        wh = CreateFileW(dio::widen(argv[1]).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        const size_t c = 8u << 20;
        std::vector<uint8_t> b(c);
        const auto w0 = std::chrono::steady_clock::now();
        // warm_at (argv 14): where the warm read starts, GiB; warm_close (argv 15): 1 = close the handle after it
        const uint64_t w_at = argc > 14 ? (uint64_t) (std::atof(argv[14]) * 1073741824.0) : 0;
        uint64_t at = w_at;
        for (; at < w_at + (uint64_t) (std::atof(argv[13]) * 1073741824.0); at += c) {
            OVERLAPPED o{};
            o.Offset = (DWORD) (at & 0xffffffffu);
            o.OffsetHigh = (DWORD) (at >> 32);
            DWORD got = 0;
            if (argc > 15 && (argv[15][0] == 'm' || argv[15][0] == 'M' || argv[15][0] == 'o')) continue;   // o / m / M: no buffered reads
            if (!ReadFile(wh, b.data(), (DWORD) c, &got, &o) || !got) break;
        }
        // warm_close 'm': instead, map the whole file (as GgufModel does) and touch the warm range through the view
        if (argc > 15 && (argv[15][0] == 'm' || argv[15][0] == 'M')) {   // M: then unmap and close it all
            const HANDLE mh = CreateFileMappingW(wh, nullptr, PAGE_READONLY, 0, 0, nullptr);
            const volatile uint8_t* v = mh ? (const volatile uint8_t*) MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0) : nullptr;
            const auto m0 = std::chrono::steady_clock::now();
            uint64_t sum = 0;
            if (v) for (uint64_t q = w_at; q < at; q += 4096) sum += v[q];
            std::printf("  map: touched %.1f GB through a view in %.1f s (%llu, kept mapped)\n", (at - w_at) / 1e9,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - m0).count(), (unsigned long long) (sum & 1));
            if (argv[15][0] == 'M') {
                UnmapViewOfFile((const void*) v);
                CloseHandle(mh);
                CloseHandle(wh);
                wh = INVALID_HANDLE_VALUE;
                std::printf("  map: unmapped and closed\n");
            }
        }
        const bool w_close = argc > 15 && argv[15][0] == '1';
        std::printf("  warm: %.1f GB from %.0f GiB through a buffered handle in %.1f s (%s)\n", (at - w_at) / 1e9, w_at / 1073741824.0,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(), w_close ? "closed" : "kept open");
        if (w_close) { CloseHandle(wh); wh = INVALID_HANDLE_VALUE; }
        // warm_wait (argv 16): seconds to wait before the reads (does closing let the file's cache state go?)
        if (argc > 16) std::this_thread::sleep_for(std::chrono::milliseconds((int) (std::atof(argv[16]) * 1000)));
    }
    void* regp = nullptr;
    size_t regn = 0;
    if (reg_gib > 0) {
        const size_t n = regn = (size_t) (reg_gib * 1073741824.0);
        void* p = regp = VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        const double t = (double) std::chrono::steady_clock::now().time_since_epoch().count();
        if (!p || cudaHostRegister(p, n, cudaHostRegisterPortable) != cudaSuccess) std::fprintf(stderr, "register failed\n");
        else std::printf("  registered %.0f GiB in %.1f s\n", reg_gib, ((double) std::chrono::steady_clock::now().time_since_epoch().count() - t) / 1e9);
    }
    if (h2d_gib > 0)
        h2d = std::thread([&] {
            const size_t n = (size_t) (h2d_gib * 1073741824.0), c = 64u << 20;
            void *hp = nullptr, *dp = nullptr;
            cudaStream_t st;
            cudaStreamCreate(&st);
            if (cudaMallocHost(&hp, n) != cudaSuccess || cudaMalloc(&dp, c) != cudaSuccess) { std::fprintf(stderr, "h2d alloc failed\n"); return; }
            std::memset(hp, 1, n);
            if (regp) {   // h2d from the registered region instead: random 10 MiB pieces, 16 queued per sync (the prompt's blobs)
                const size_t piece = 10u << 20;
                uint64_t x = 88172645463325252ull;
                while (!h2d_stop.load()) {
                    for (int q = 0; q < 6; ++q) {
                        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                        cudaMemcpyAsync((char*) dp + q * piece, (char*) regp + (x % ((regn - piece) >> 12) << 12), piece, cudaMemcpyHostToDevice, st);
                    }
                    cudaStreamSynchronize(st);
                    h2d_bytes += (int64_t) (6 * piece);
                }
            }
            while (!h2d_stop.load()) {
                for (size_t at = 0; at + c <= n && !h2d_stop.load(); at += c) {
                    cudaMemcpyAsync(dp, (char*) hp + at, c, cudaMemcpyHostToDevice, st);
                    cudaStreamSynchronize(st);
                    h2d_bytes += (int64_t) c;
                }
            }
            cudaFreeHost(hp);
            cudaFree(dp);
        });
#endif
    if (h2d.joinable()) std::this_thread::sleep_for(std::chrono::seconds(3));   // pinning + first copies
    const int64_t h2d0 = h2d_bytes.load();
    std::atomic<int64_t> next{parts ? first * std::max(1, std::atoi(argv[6])) : first}, bytes{0}, fails{0};
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for (int k = 0; k < nth; ++k)
        th.emplace_back([&] {
            uint8_t* sc = (uint8_t*) dio::aligned_alloc(4096, 3 * bounce);
            uint8_t* dstp = (uint8_t*) dio::aligned_alloc(4096, 3 * bounce);
            struct Dst { uint8_t* p; uint8_t* data() { return p; } } dst{dstp};
            std::vector<OVERLAPPED> ov;
            std::vector<HANDLE> ev;
            if (parts) {   // Ds41FileBlobs::read_part, P parts a blob, tasks = blob * P + part
                for (int64_t x; (x = next.fetch_add(1)) < (first + count) * P;) {
                    if (x < first * P) continue;
                    const int64_t b = x / P, part = x % P;
                    const Span* s = &sp[(size_t) (b * 3)];
                    const uint64_t total = s[0].bytes + s[1].bytes + s[2].bytes;
                    uint64_t lo = total * (uint64_t) part / (uint64_t) P, hi = total * (uint64_t) (part + 1) / (uint64_t) P;
                    lo &= ~4095ull;
                    if (part + 1 < P) hi &= ~4095ull; else hi = total;
                    uint64_t at = 0, n = 0;
                    for (int r = 0; r < 3; ++r) {
                        const uint64_t a = std::max(lo, at), e = std::min(hi, at + s[r].bytes);
                        if (a < e) {
                            const uint64_t off = s[r].off + (a - at);
                            const bool ip = inplace && (off % 512) == 0 && ((e - a) % 512) == 0 && (a % 512) == 0;
                            const uint64_t a0 = ip ? off : off & ~4095ull, a1 = ip ? off + (e - a) : (off + (e - a) + 4095) & ~4095ull;
                            uint8_t* to = ip ? dst.data() + a : sc;
                            uint64_t got = 0;
                            while (got < a1 - a0) {
                                const ssize_t k = dio::pread(fd, to + got, (size_t) (a1 - a0 - got), (int64_t) (a0 + got));
                                if (k <= 0) { fails.fetch_add(1); break; }
                                got += (uint64_t) k;
                            }
                            if (copy && !ip) std::memcpy(dst.data() + a, sc + (off - a0), (size_t) (e - a));
                            n += e - a;
                        }
                        at += s[r].bytes;
                    }
                    bytes.fetch_add((int64_t) n);
                }
            }
            for (int64_t b; !parts && (b = next.fetch_add(1)) < first + count;) {
                uint64_t a0[3], a1[3];
                for (int r = 0; r < 3; ++r) {
                    const Span& s = sp[(size_t) (b * 3 + r)];
                    a0[r] = s.off & ~4095ull;
                    a1[r] = (s.off + s.bytes + 4095) & ~4095ull;
                }
                if (!split) {
                    for (int r = 0; r < 3; ++r) {
                        uint64_t got = 0;
                        while (got < a1[r] - a0[r]) {
                            const ssize_t n = dio::pread(fd, sc + r * bounce + got, (size_t) (a1[r] - a0[r] - got), (int64_t) (a0[r] + got));
                            if (n <= 0) { fails.fetch_add(1); break; }
                            got += (uint64_t) n;
                        }
                    }
                } else {
                    size_t q = 0;
                    for (int r = 0; r < 3; ++r) q += (size_t) ((a1[r] - a0[r] + chunk - 1) / chunk);
                    while (ev.size() < q) ev.push_back(CreateEventW(nullptr, TRUE, FALSE, nullptr));
                    ov.assign(q, OVERLAPPED{});
                    size_t i = 0;
                    for (int r = 0; r < 3; ++r)
                        for (uint64_t at = a0[r]; at < a1[r]; at += chunk, ++i) {
                            const DWORD want = (DWORD) std::min<uint64_t>(chunk, a1[r] - at);
                            ov[i].Offset = (DWORD) (at & 0xffffffffu);
                            ov[i].OffsetHigh = (DWORD) (at >> 32);
                            ov[i].hEvent = ev[i];
                            ResetEvent(ev[i]);
                            if (!ReadFile(h, sc + r * bounce + (at - a0[r]), want, nullptr, &ov[i]) && GetLastError() != ERROR_IO_PENDING)
                                fails.fetch_add(1);
                        }
                    for (size_t j = 0; j < q; ++j) {
                        DWORD got = 0;
                        if (!GetOverlappedResult(h, &ov[j], &got, TRUE)) fails.fetch_add(1);
                    }
                }
                uint64_t n = 0;
                for (int r = 0; r < 3; ++r) {
                    const Span& s = sp[(size_t) (b * 3 + r)];
                    if (copy) std::memcpy(dst.data() + r * bounce, sc + r * bounce + (s.off - a0[r]), (size_t) s.bytes);
                    n += s.bytes;
                }
                bytes.fetch_add((int64_t) n);
            }
            for (HANDLE e : ev) CloseHandle(e);
            dio::aligned_free(dstp);
            dio::aligned_free(sc);
        });
    for (auto& t : th) t.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    {
        FILETIME c, e, k, u;
        GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
        std::printf("  process cpu: kernel %.0f ms, user %.0f ms%c", ((uint64_t) k.dwHighDateTime << 32 | k.dwLowDateTime) / 10000.0,
                    ((uint64_t) u.dwHighDateTime << 32 | u.dwLowDateTime) / 10000.0, 10);
    }
    const int64_t h2d1 = h2d_bytes.load();
    h2d_stop = true;
    if (h2d.joinable()) {
        h2d.join();
        std::printf("  h2d alongside: %.2f GB/s\n", (h2d1 - h2d0) / 1e9 / s);
    }
    std::printf("%s %s thr=%d chunk=%lluK flags=%c blobs=%lld..%lld: %.2f GB in %.2f s = %.2f GB/s%s\n", argv[3], argv[4],
                nth, (unsigned long long) (chunk / 1024), fl, (long long) first, (long long) (first + count),
                bytes.load() / 1e9, s, bytes.load() / 1e9 / s, fails.load() ? "  (READ FAILURES)" : "");
    dio::close(fd);
    return 0;
#endif
}
