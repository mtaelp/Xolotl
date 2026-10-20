// tools/v41/engine/ds41_gpu2.cpp - --gpu2: the second card's helper process and the engine's side of it (see the .hpp).
#include "ds41_gpu2.hpp"

#include "ds41_moe.hpp"
#include "ds41_transcode.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(DS4_MOE_CUDA)
#include <cuda_runtime.h>
#endif

namespace strata::v41 {

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr uint32_t kMagic = 0x32555047;   // "GPU2"
constexpr uint32_t kVersion = 2;

/// The shared memory's header.  The buffers follow at 64 KiB-aligned offsets (layout()).
struct Hdr {
    uint32_t magic, version;
    int64_t n_layers, n_experts, top_k, n_embd;
    int32_t max_rows, piece, chunk_mmq, pad0;
    double reserve_gib, max_gib, skip_gib;
    char model[2048], rank[2048];
    // worker -> engine once, after its seed
    volatile int32_t status;   ///< 0 starting, 1 ready, -1 failed (msg says why)
    int32_t pad1;
    int64_t owned_n;
    double owned_gib, seed_s, free_gib;
    char msg[2048];
    // per call
    volatile int64_t req, done;
    int64_t layer;
    int32_t op, nrows;
    volatile int32_t result;   ///< 1 = the call ran, 0 = the worker's tier refused it
    int32_t pad2;
    double worker_ms;
};

struct Layout {
    size_t owned, ids, w, x, out, total;
};

size_t up64k(size_t v) { return (v + 65535) & ~(size_t) 65535; }

Layout layout(int64_t n_layers, int64_t n_experts, int64_t top_k, int64_t n_embd, int64_t rows) {
    Layout L;
    L.owned = up64k(sizeof(Hdr));
    L.ids = L.owned + up64k((size_t) (n_layers * n_experts));
    L.w = L.ids + up64k((size_t) (rows * top_k) * 4);
    L.x = L.w + up64k((size_t) (rows * top_k) * 4);
    L.out = L.x + up64k((size_t) (rows * n_embd) * 4);
    L.total = L.out + up64k((size_t) (rows * n_embd) * 4);
    return L;
}

}  // namespace

Ds41Gpu2::Ds41Gpu2() = default;

#if defined(_WIN32)

Ds41Gpu2::~Ds41Gpu2() {
    if (base_ && proc_ && ok_) {   // ask the worker to go, then make sure it does
        Hdr* h = (Hdr*) base_;
        if (pending_) WaitForSingleObject((HANDLE) ev_done_, 2000);
        h->op = kQuit;
        InterlockedIncrement64((volatile LONG64*) &h->req);
        SetEvent((HANDLE) ev_req_);
        WaitForSingleObject((HANDLE) proc_, 3000);
    }
    if (proc_) { TerminateProcess((HANDLE) proc_, 0); CloseHandle((HANDLE) proc_); }
    if (job_) CloseHandle((HANDLE) job_);
    if (base_) UnmapViewOfFile(base_);
    if (map_) CloseHandle((HANDLE) map_);
    if (ev_req_) CloseHandle((HANDLE) ev_req_);
    if (ev_done_) CloseHandle((HANDLE) ev_done_);
}

void Ds41Gpu2::fail(const std::string& why) {
    if (ok_) std::fprintf(stderr, "gpu2: %s - the main card computes every expert from here on\n", why.c_str());
    ok_ = false;
    pending_ = false;
}

bool Ds41Gpu2::start(const Ds41Gpu2Config& c, std::string& err) {
    c_ = c;
    if (c_.n_layers <= 0 || c_.n_experts <= 0 || c_.top_k <= 0 || c_.n_embd <= 0 || c_.max_rows <= 0) {
        err = "gpu2: the geometry is not set";
        return false;
    }
    c_.piece = std::max(1, std::min(c_.piece, c_.max_rows));
    const Layout L = layout(c_.n_layers, c_.n_experts, c_.top_k, c_.n_embd, c_.max_rows);
    const std::string name = "Local\\ds41_gpu2_" + std::to_string(GetCurrentProcessId());
    map_ = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD) ((uint64_t) L.total >> 32),
                              (DWORD) L.total, name.c_str());
    if (!map_) { err = "gpu2: CreateFileMapping failed (" + std::to_string(GetLastError()) + ")"; return false; }
    base_ = (uint8_t*) MapViewOfFile((HANDLE) map_, FILE_MAP_ALL_ACCESS, 0, 0, L.total);
    if (!base_) { err = "gpu2: MapViewOfFile failed (" + std::to_string(GetLastError()) + ")"; return false; }
    ev_req_ = CreateEventA(nullptr, FALSE, FALSE, (name + "_req").c_str());
    ev_done_ = CreateEventA(nullptr, FALSE, FALSE, (name + "_done").c_str());
    if (!ev_req_ || !ev_done_) { err = "gpu2: CreateEvent failed"; return false; }
    Hdr* h = (Hdr*) base_;
    std::memset(h, 0, sizeof(Hdr));
    h->magic = kMagic;
    h->version = kVersion;
    h->n_layers = c_.n_layers;
    h->n_experts = c_.n_experts;
    h->top_k = c_.top_k;
    h->n_embd = c_.n_embd;
    h->max_rows = c_.max_rows;
    h->piece = c_.piece;
    h->chunk_mmq = c_.chunk_mmq ? 1 : 0;
    h->reserve_gib = c_.reserve_gib;
    h->max_gib = c_.max_gib;
    h->skip_gib = c_.skip_gib;
    std::snprintf(h->model, sizeof(h->model), "%s", c_.model.c_str());
    std::snprintf(h->rank, sizeof(h->rank), "%s", c_.rank.c_str());
    ids_ = (int32_t*) (base_ + L.ids);
    w_ = (float*) (base_ + L.w);
    x_ = (float*) (base_ + L.x);
    out_ = (float*) (base_ + L.out);
    return spawn(err);
}

bool Ds41Gpu2::restart(std::string& err) {
    if (!base_) { err = "gpu2: not started"; return false; }
    if (proc_) {
        TerminateProcess((HANDLE) proc_, 0);
        WaitForSingleObject((HANDLE) proc_, 5000);
        CloseHandle((HANDLE) proc_);
        proc_ = nullptr;
    }
    Hdr* h = (Hdr*) base_;
    h->status = 0;
    h->req = 0;
    h->done = 0;
    req_ = 0;
    pending_ = false;
    ResetEvent((HANDLE) ev_req_);
    ResetEvent((HANDLE) ev_done_);
    ++restarts;
    return spawn(err) && wait_ready(err);
}

bool Ds41Gpu2::spawn(std::string& err) {
    const std::string name = "Local\\ds41_gpu2_" + std::to_string(GetCurrentProcessId());
    // the worker: this executable, its stdout and stderr on our stderr (stdout is the server's protocol pipe), only
    // the second card visible, no RAM arena of its own (STRATA_ARENA_REBAL=0) and small streaming halves (it never
    // streams: every expert it is sent sits in its VRAM)
    char exe[MAX_PATH * 2] = "";
    GetModuleFileNameA(nullptr, exe, sizeof(exe));
    std::string cmd = std::string("\"") + exe + "\" --gpu2-worker " + name + " " + std::to_string(GetCurrentProcessId());
    struct Env { std::string k; std::string v; std::string old; bool had; };
    std::vector<Env> env = { {"CUDA_DEVICE_ORDER", "PCI_BUS_ID", "", false}, {"CUDA_VISIBLE_DEVICES", std::to_string(c_.gpu), "", false},
                             {"STRATA_ARENA_REBAL", "0", "", false}, {"MIMO_CHUNK_STAGE_MIB", "256", "", false},
                             {"STRATA_MMQ_TC_KEEP_GIB", "0.25", "", false}, {"STRATA_MMQ_TC_GIB", "0.5", "", false} };
    // STRATA_GPU2_ENV="K=V;K=V": settings for the worker alone (profiling it: DS41_PROF_X=2;MIMO_CHUNK_PROF=1)
    if (const char* xe = std::getenv("STRATA_GPU2_ENV")) {
        std::string all = xe;
        for (size_t at = 0; at < all.size();) {
            size_t semi = all.find(';', at);
            if (semi == std::string::npos) semi = all.size();
            const std::string kv = all.substr(at, semi - at);
            const size_t eq = kv.find('=');
            if (eq != std::string::npos && eq > 0) env.push_back({kv.substr(0, eq), kv.substr(eq + 1), "", false});
            at = semi + 1;
        }
    }
    for (Env& e : env) {
        char buf[4096];
        const DWORD n = GetEnvironmentVariableA(e.k.c_str(), buf, sizeof(buf));
        e.had = n > 0 && n < sizeof(buf);
        if (e.had) e.old = buf;
        SetEnvironmentVariableA(e.k.c_str(), e.v.c_str());
    }
    HANDLE herr = GetStdHandle(STD_ERROR_HANDLE);
    HANDLE hdup = nullptr;
    if (herr && herr != INVALID_HANDLE_VALUE)
        DuplicateHandle(GetCurrentProcess(), herr, GetCurrentProcess(), &hdup, 0, TRUE, DUPLICATE_SAME_ACCESS);
    SIZE_T asz = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &asz);
    std::vector<uint8_t> abuf(asz);
    auto* al = (LPPROC_THREAD_ATTRIBUTE_LIST) abuf.data();
    bool attr = InitializeProcThreadAttributeList(al, 1, 0, &asz) != FALSE;
    if (attr && hdup) attr = UpdateProcThreadAttribute(al, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, &hdup, sizeof(HANDLE), nullptr, nullptr) != FALSE;
    STARTUPINFOEXA si{};
    si.StartupInfo.cb = sizeof(si);
    if (hdup) {
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        si.StartupInfo.hStdInput = nullptr;
        si.StartupInfo.hStdOutput = hdup;
        si.StartupInfo.hStdError = hdup;
    }
    si.lpAttributeList = attr && hdup ? al : nullptr;
    PROCESS_INFORMATION pi{};
    std::vector<char> cl(cmd.begin(), cmd.end());
    cl.push_back(0);
    const BOOL made = CreateProcessA(nullptr, cl.data(), nullptr, nullptr, hdup && attr ? TRUE : FALSE,
                                     CREATE_SUSPENDED | CREATE_NO_WINDOW | (si.lpAttributeList ? EXTENDED_STARTUPINFO_PRESENT : 0),
                                     nullptr, nullptr, (LPSTARTUPINFOA) &si, &pi);
    const DWORD cerr = GetLastError();
    if (attr) DeleteProcThreadAttributeList(al);
    if (hdup) CloseHandle(hdup);
    for (Env& e : env) SetEnvironmentVariableA(e.k.c_str(), e.had ? e.old.c_str() : nullptr);
    if (!made) { err = "gpu2: cannot start the worker (CreateProcess " + std::to_string(cerr) + ")"; return false; }
    // the worker ends with the engine, however the engine ends
    if (!job_) job_ = CreateJobObjectA(nullptr, nullptr);
    if (job_) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION jl{};
        jl.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject((HANDLE) job_, JobObjectExtendedLimitInformation, &jl, sizeof(jl));
        AssignProcessToJobObject((HANDLE) job_, pi.hProcess);
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    proc_ = pi.hProcess;
    return true;
}

bool Ds41Gpu2::wait_ready(std::string& err) {
    Hdr* h = (Hdr*) base_;
    if (!h || !proc_) { err = "gpu2: not started"; return false; }
    for (;;) {
        if (h->status != 0) break;
        HANDLE hs[2] = { (HANDLE) ev_done_, (HANDLE) proc_ };
        const DWORD r = WaitForMultipleObjects(2, hs, FALSE, 1000);
        if (r == WAIT_OBJECT_0 + 1 && h->status == 0) { err = "gpu2: the worker exited before it was ready"; return false; }
    }
    if (h->status < 0) { err = std::string("gpu2: ") + h->msg; return false; }
    const Layout L = layout(c_.n_layers, c_.n_experts, c_.top_k, c_.n_embd, c_.max_rows);
    owned_.assign(base_ + L.owned, base_ + L.owned + (size_t) (c_.n_layers * c_.n_experts));
    owned_n_ = h->owned_n;
    owned_gib_ = h->owned_gib;
    report_ = h->msg;
    ok_ = owned_n_ > 0;
    if (!ok_) { err = "gpu2: the worker holds no experts (" + report_ + ")"; return false; }
    return true;
}

bool Ds41Gpu2::submit(Op op, int64_t layer, int nrows) {
    if (!ok_ || pending_) return false;
    Hdr* h = (Hdr*) base_;
    h->op = op;
    h->layer = layer;
    h->nrows = nrows;
    h->result = 0;
    req_ = InterlockedIncrement64((volatile LONG64*) &h->req);
    SetEvent((HANDLE) ev_req_);
    pending_ = true;
    kind_ = op == kChunk ? 1 : 0;
    ++calls_k[kind_];
    ++calls;
    rows += nrows;
    return true;
}

bool Ds41Gpu2::wait(std::string& err) {
    if (!pending_) { err = "gpu2: no call pending"; return false; }
    Hdr* h = (Hdr*) base_;
    const double t0 = now_ms();
    // the worker usually finishes during the caller's own work: check first, block only if it has not
    while (h->done != req_ && now_ms() - t0 < 2.0)
        for (int i = 0; i < 64; ++i) YieldProcessor();
    while (h->done != req_) {
        HANDLE hs[2] = { (HANDLE) ev_done_, (HANDLE) proc_ };
        const DWORD r = WaitForMultipleObjects(2, hs, FALSE, 5000);
        if (r == WAIT_OBJECT_0 + 1 && h->done != req_) { fail("the worker exited"); err = "gpu2: the worker exited"; return false; }
        if (r == WAIT_TIMEOUT && h->done != req_ && now_ms() - t0 > 120000) {
            fail("the worker did not answer for 120 s");
            err = "gpu2: worker timeout";
            return false;
        }
    }
    MemoryBarrier();
    wait_ms += now_ms() - t0;
    worker_ms += h->worker_ms;
    wait_k[kind_] += now_ms() - t0;
    worker_k[kind_] += h->worker_ms;
    pending_ = false;
    if (!h->result) { fail(std::string("the worker refused a call: ") + h->msg); err = "gpu2: worker call failed"; return false; }
    return true;
}

int ds41_gpu2_worker_main(const char* name, unsigned long parent_pid) {
    HANDLE map = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
    if (!map) { std::fprintf(stderr, "gpu2 worker: cannot open %s\n", name); return 1; }
    uint8_t* base = (uint8_t*) MapViewOfFile(map, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    HANDLE ev_req = OpenEventA(SYNCHRONIZE | EVENT_MODIFY_STATE, FALSE, (std::string(name) + "_req").c_str());
    HANDLE ev_done = OpenEventA(SYNCHRONIZE | EVENT_MODIFY_STATE, FALSE, (std::string(name) + "_done").c_str());
    HANDLE parent = OpenProcess(SYNCHRONIZE, FALSE, (DWORD) parent_pid);
    if (!base || !ev_req || !ev_done) { std::fprintf(stderr, "gpu2 worker: cannot attach to %s\n", name); return 1; }
    Hdr* h = (Hdr*) base;
    auto fail = [&](const std::string& m) {
        std::snprintf(h->msg, sizeof(h->msg), "%s", m.c_str());
        MemoryBarrier();
        h->status = -1;
        SetEvent(ev_done);
        std::fprintf(stderr, "gpu2 worker: %s\n", m.c_str());
        return 1;
    };
    if (h->magic != kMagic || h->version != kVersion) return fail("shared memory version mismatch");
    const Layout L = layout(h->n_layers, h->n_experts, h->top_k, h->n_embd, h->max_rows);
    const int64_t NL = h->n_layers, NX = h->n_experts, K = h->top_k, H = h->n_embd;
#if !defined(DS4_MOE_CUDA)
    return fail("this build has no CUDA tier");
#else
    const double t0 = now_ms();
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return fail("no CUDA device visible (CUDA_VISIBLE_DEVICES)");
    Ds41FileBlobs src(h->model);
    std::string err;
    if (!src.open(err)) return fail("experts: " + err);
    const Ds41MoeGeom& G = src.geom();
    if (G.n_layers != NL || G.n_experts != NX || G.top_k != K || G.n_embd != H) return fail("the model's geometry differs from the engine's");
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (h->rank[0]) {
        ranked = ds41_read_ranked(h->rank, err);
        if (ranked.empty()) return fail("--rank: " + err);
    } else {
        for (int64_t l = 0; l < NL; ++l)
            for (int64_t e = 0; e < NX; ++e) ranked.emplace_back((int32_t) l, (int32_t) e);
    }
    // skip_gib: the head of the ranking stays the main card's (its decode cache); this card takes what follows
    double skipped = 0;
    int64_t n_skip = 0;
    if (h->skip_gib > 0) {
        std::vector<char> seen((size_t) (NL * NX), 0);
        size_t i = 0;
        for (; i < ranked.size(); ++i) {
            const auto& le = ranked[i];
            if (le.first < 0 || le.first >= NL || le.second < 0 || le.second >= NX || !G.routed(le.first)) continue;
            char& sn = seen[(size_t) (le.first * NX + le.second)];
            if (sn) continue;
            const double b = (double) src.expert_bytes(le.first, le.second);
            if (skipped + b > h->skip_gib * 1073741824.0) break;
            sn = 1;
            skipped += b;
            ++n_skip;
        }
        ranked.erase(ranked.begin(), ranked.begin() + (std::ptrdiff_t) i);
    }
    Ds41MoeConfig mc;
    mc.slots = NL * NX;
    mc.arena_gib = 0;
    mc.max_arena_gib = 0;
    mc.pin = false;
    mc.pf_b = 0;
    mc.vram_lru = false;
    mc.arena_adapt = false;
    mc.chunk_mmq = h->chunk_mmq != 0;
    mc.chunk_resident_mmq = true;   // its experts are all resident: MMQ for them too (ds41_moe.hpp)
    mc.chunk_prefetch = false;
    mc.threads = 2;            // the CPU pool is never used (every expert it is sent is in VRAM)
    mc.tier_batch_pcie = 1.0;
    Ds41MoeTier tier;
    if (!tier.init(G, &src, mc, err)) return fail("tier init: " + err);
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    double gib = (double) fr / 1073741824.0 - h->reserve_gib;
    if (h->max_gib > 0) gib = std::min(gib, h->max_gib);
    if (gib < 0.5) return fail("only " + std::to_string(fr / 1073741824.0) + " GiB of VRAM free on " + prop.name);
    tier.set_slot_gib(gib);
    if (!tier.seed_from_ranked(ranked, err)) return fail("seed: " + err);
    uint8_t* owned = base + L.owned;
    int64_t n = 0;
    double bytes = 0;
    for (int64_t l = 0; l < NL; ++l)
        for (int64_t e = 0; e < NX; ++e) {
            const bool r = tier.resident(l, e);
            owned[(size_t) (l * NX + e)] = r ? 1 : 0;
            if (r) { ++n; bytes += (double) src.expert_bytes(l, e); }
        }
    // one expert through the kernels before saying ready: a build without this card's architecture (CUDA_ARCH) fails
    // here, not on the first request
    if (n > 0) {
        int64_t tl = -1, te = -1;
        for (int64_t i = 0; i < NL * NX && tl < 0; ++i)
            if (owned[(size_t) i]) { tl = i / NX; te = i % NX; }
        std::vector<int32_t> tid((size_t) K, -1);
        std::vector<float> tw((size_t) K, 0.0f), tx((size_t) H, 0.01f), to((size_t) H, -1.0f);
        tid[0] = (int32_t) te;
        tw[0] = 1.0f;
        bool tok = tier.run_multi(tl, 1, tid.data(), tw.data(), tx.data(), to.data()) && cudaDeviceSynchronize() == cudaSuccess;
        for (int64_t i = 0; tok && i < H; ++i) tok = std::isfinite(to[(size_t) i]);
        if (!tok) {
            const cudaError_t ce = cudaGetLastError();
            return fail(std::string("kernel test on ") + prop.name + " (sm_" + std::to_string(prop.major) + std::to_string(prop.minor) +
                        ") failed: " + cudaGetErrorString(ce) + " - is the engine built for this card? (CUDA_ARCH, build-ds41-windows.bat)");
        }
        tier.reset_stats();
    }
    size_t fr2 = 0;
    cudaMemGetInfo(&fr2, &tot);
    h->owned_n = n;
    h->owned_gib = bytes / 1073741824.0;
    h->seed_s = (now_ms() - t0) / 1000.0;
    h->free_gib = fr2 / 1073741824.0;
    std::snprintf(h->msg, sizeof(h->msg), "%s (sm_%d%d): %lld experts in VRAM (%.2f GiB, after the ranking's first %lld = %.2f GiB), "
                  "%.2f GiB free, seeded in %.1f s", prop.name, prop.major, prop.minor, (long long) n, h->owned_gib,
                  (long long) n_skip, skipped / 1073741824.0, h->free_gib, h->seed_s);
    std::fprintf(stderr, "gpu2 worker: %s\n", h->msg);
    tier.reset_stats();
    MemoryBarrier();
    h->status = 1;
    SetEvent(ev_done);

    // STRATA_GPU2_WARM (default 1): between calls, while they keep coming (one in the last 200 ms), this loop runs
    // 50 us spin kernels on the card instead of idling.  A decode pass gives this card ~0.1-0.3 ms of work per layer
    // call, too little for the driver to keep its clocks up (measured: ~700 MHz core, half the memory clock, 1.1-1.3
    // ms a call; warm: 0.46 ms).  The spins run here, between calls, never next to the tier's kernels: a spin on its
    // own thread and stream beside them made the tier's grouped kernels fail (unspecified launch failure, 4 times
    // in 4 server runs).
    static const bool warm_on = !(std::getenv("STRATA_GPU2_WARM") && std::getenv("STRATA_GPU2_WARM")[0] == '0');
    int khz = 0;
    if (cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0) != cudaSuccess || khz <= 0) khz = 2000000;
    cudaStream_t ws = nullptr;
    if (warm_on && cudaStreamCreateWithFlags(&ws, cudaStreamNonBlocking) != cudaSuccess) ws = nullptr;
    int32_t* ids = (int32_t*) (base + L.ids);
    float* w = (float*) (base + L.w);
    float* x = (float*) (base + L.x);
    float* out = (float*) (base + L.out);
    int64_t seen = 0;
    // STRATA_GPU2_SPIN_US (default 3000): after a call, poll for the next one this long before sleeping on the event - a
    // decode / verify pass sends one call per layer ~2-3 ms apart, and waking a sleeping thread costs more than that
    static const double spin_ms = (std::getenv("STRATA_GPU2_SPIN_US") ? std::atof(std::getenv("STRATA_GPU2_SPIN_US")) : 3000.0) / 1000.0;
    double last = now_ms();
    for (;;) {
        while (h->req == seen && now_ms() - last < (ws ? 200.0 : spin_ms)) {
            if (ws) {
                ds41_spin_us(50.0, khz, ws);
                cudaStreamSynchronize(ws);
            } else {
                for (int i = 0; i < 64; ++i) YieldProcessor();
            }
        }
        if (h->req == seen) {
            HANDLE hs[2] = { ev_req, parent ? parent : ev_req };
            const DWORD r = WaitForMultipleObjects(parent ? 2 : 1, hs, FALSE, 2000);
            if (r == WAIT_OBJECT_0 + 1) return 0;   // the engine is gone
        }
        while (h->req != seen) {
            seen = h->req;
            MemoryBarrier();
            if (h->op == Ds41Gpu2::kQuit) return 0;
            const double tc = now_ms();
            const int nr = h->nrows;
            bool ok = nr >= 0 && nr <= h->max_rows;
            if (ok && h->op == Ds41Gpu2::kMulti) {
                for (int r0 = 0; ok && r0 < nr; r0 += Ds41MoeTier::kMaxMulti) {
                    const int m = std::min(Ds41MoeTier::kMaxMulti, nr - r0);
                    ok = tier.run_multi(h->layer, m, ids + (size_t) r0 * K, w + (size_t) r0 * K, x + (size_t) r0 * H,
                                        out + (size_t) r0 * H);
                }
            } else if (ok && h->op == Ds41Gpu2::kChunk) {
                for (int r0 = 0; ok && r0 < nr; r0 += h->piece) {
                    const int m = std::min(h->piece, nr - r0);
                    ok = tier.run_chunk(h->layer, m, ids + (size_t) r0 * K, w + (size_t) r0 * K, x + (size_t) r0 * H,
                                        out + (size_t) r0 * H);
                }
            } else {
                ok = false;
            }
            if (!ok) std::snprintf(h->msg, sizeof(h->msg), "op %d layer %lld rows %d refused", h->op, (long long) h->layer, nr);
            static const bool cprof = std::getenv("MIMO_CHUNK_PROF") != nullptr;
            if (cprof && h->op == Ds41Gpu2::kChunk) {
                std::fprintf(stderr, "gpu2 worker: layer %lld, %d rows in %.1f ms\n", (long long) h->layer, nr, now_ms() - tc);
                if (h->layer == 42 || h->layer == NL - 1) tier.release_chunk();   // prints the chunk profile
            }
            h->worker_ms = now_ms() - tc;
            h->result = ok ? 1 : 0;
            MemoryBarrier();
            h->done = seen;
            SetEvent(ev_done);
            last = now_ms();
        }
    }
#endif
}

#else   // not Windows: the second process exists for WDDM's pinning limit; elsewhere --gpu2 is not built

Ds41Gpu2::~Ds41Gpu2() = default;
void Ds41Gpu2::fail(const std::string& why) { (void) why; ok_ = false; }
bool Ds41Gpu2::start(const Ds41Gpu2Config& c, std::string& err) {
    c_ = c;
    err = "--gpu2 is Windows-only (a helper process around WDDM's pinned-memory limit)";
    return false;
}
bool Ds41Gpu2::wait_ready(std::string& err) { err = "--gpu2 is Windows-only"; return false; }
bool Ds41Gpu2::restart(std::string& err) { err = "--gpu2 is Windows-only"; return false; }
bool Ds41Gpu2::spawn(std::string& err) { err = "--gpu2 is Windows-only"; return false; }
bool Ds41Gpu2::submit(Op, int64_t, int) { return false; }
bool Ds41Gpu2::wait(std::string& err) { err = "--gpu2 is Windows-only"; return false; }
int ds41_gpu2_worker_main(const char* name, unsigned long parent_pid) {
    (void) name;
    (void) parent_pid;
    std::fprintf(stderr, "--gpu2-worker is Windows-only\n");
    return 1;
}

#endif

}  // namespace strata::v41
