// tools/v41/engine/ds41_gpu2.hpp - --gpu2: a second NVIDIA card as a VRAM expert cache next to the main one.
//
// WHY A SECOND PROCESS.  On Windows (WDDM) one process that opens both cards cannot pin more than ~60 GiB of host RAM:
// measured on an RTX 5090 + RTX 3090, 60 GiB of cudaHostRegister'd RAM left both cards their whole VRAM, 64 GiB took
// it all away (the 3090 got 0 GiB, the 5090 6.5).  The engine's RAM arena is 96 GiB.  The same 96 GiB pinned in a
// process that opens only the 5090, plus a second process that opens only the 3090, leaves both cards all of their
// VRAM at full speed.  So the second card lives in a helper process (`ds41_generate --gpu2-worker`, started by the
// engine) and the two talk through shared memory.  docs/WINDOWS_DS41.md "Dual GPU" has the measurements.
//
// WHAT IT DOES.  The worker opens the model's experts itself, fills its VRAM with the head of the routing rank
// (--rank) and says which experts it holds.  The engine's tier leaves those out of its own VRAM cache and RAM arena
// (Ds41MoeConfig::exclude), so the two cards plus the arena hold ~22 GiB more experts than one card does, and fewer
// are read from the SSD.  Per layer call, the tier sends the rows that route to a worker expert (their activations,
// the worker's ids and weights; -1 / 0 for the rest) and computes its own experts meanwhile; the worker's partial sums
// come back through the same shared memory and are added on the main card.  Only activations cross the second card's
// link (PCIe 4.0 x4 over OCuLink here: 6.7 GB/s, ~40 us for a small round trip).
//
// Windows only (the WDDM limit is the reason it exists).  Elsewhere `start` fails with a message and the engine runs
// on one card.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::v41 {

struct Ds41Gpu2Config {
    std::string model;          ///< the GGUF (the worker reads its experts from it)
    std::string rank;           ///< --rank: the worker takes the head of this ranking (empty = layer/expert order)
    int gpu = 1;                ///< the second card as nvidia-smi numbers it (CUDA_DEVICE_ORDER=PCI_BUS_ID)
    double reserve_gib = 3.0;   ///< VRAM the worker keeps free for its own buffers (chunk rows, MMQ, the desktop)
    double max_gib = 0.0;       ///< cap on the worker's expert slots, 0 = whatever is free minus reserve_gib
    /// Skip this many GiB of the ranking's head before the worker takes its experts: the head is what the main card's
    /// own VRAM cache holds while decoding, so the second card holds the NEXT experts (the main card's decode misses).
    double skip_gib = 0.0;
    int max_rows = 12288;       ///< the most rows one call carries (the prompt batch); a larger call goes in pieces
    int piece = 4096;           ///< rows per worker run_chunk (its chunk buffers scale with this)
    bool chunk_mmq = true;      ///< the worker's prompt rows on MMQ, as the engine's (--chunk-mmq)
    int64_t n_layers = 0, n_experts = 0, top_k = 0, n_embd = 0;
};

class Ds41Gpu2 {
public:
    enum Op : int32_t { kMulti = 1, kChunk = 2, kQuit = 3 };

    Ds41Gpu2();
    ~Ds41Gpu2();
    Ds41Gpu2(const Ds41Gpu2&) = delete;
    Ds41Gpu2& operator=(const Ds41Gpu2&) = delete;

    /// Starts the worker; returns at once.  `wait_ready` then blocks until it has seeded its VRAM (or failed).
    bool start(const Ds41Gpu2Config& c, std::string& err);
    bool wait_ready(std::string& err);
    /// After a failure (ok() false): a fresh worker, seeded again (~5 s).  The engine calls it between requests.
    bool restart(std::string& err);
    /// Alive and answering (false after a failed call: the tier then computes everything itself).
    bool ok() const { return ok_; }
    bool owns(int64_t layer, int64_t expert) const {
        return ok_ && layer >= 0 && expert >= 0 && expert < c_.n_experts && layer < c_.n_layers &&
               owned_[(size_t) (layer * c_.n_experts + expert)] != 0;
    }
    /// [n_layers * n_experts], 1 = the worker holds it
    const std::vector<char>& owned() const { return owned_; }
    int64_t owned_count() const { return owned_n_; }
    double owned_gib() const { return owned_gib_; }
    const std::string& report() const { return report_; }
    int max_rows() const { return c_.max_rows; }
    const Ds41Gpu2Config& config() const { return c_; }

    /// One call's buffers in the shared memory (max_rows rows): ids / w token-major [rows * top_k] (-1 / 0 where the
    /// worker does not hold the expert), x [rows * n_embd] in, out [rows * n_embd] back.
    int32_t* ids() const { return ids_; }
    float* w() const { return w_; }
    float* x() const { return x_; }
    float* out() const { return out_; }
    size_t out_bytes() const { return (size_t) c_.max_rows * (size_t) c_.n_embd * 4; }
    /// Hands the filled buffers to the worker (asynchronous); `wait` blocks until its sums are in out().
    bool submit(Op op, int64_t layer, int nrows);
    bool wait(std::string& err);

    /// Calls / rows / entries / ms waited (after the caller's own work) since start.
    int64_t calls = 0, rows = 0, entries = 0, restarts = 0;
    double wait_ms = 0, worker_ms = 0;
    /// the same split by kind: [0] decode / verify calls (kMulti), [1] prompt chunks (kChunk)
    int64_t calls_k[2] = {0, 0};
    double wait_k[2] = {0, 0}, worker_k[2] = {0, 0};

private:
    bool spawn(std::string& err);
    void fail(const std::string& why);
    Ds41Gpu2Config c_;
    bool ok_ = false, pending_ = false;
    int kind_ = 0;
    std::vector<char> owned_;
    int64_t owned_n_ = 0;
    double owned_gib_ = 0;
    std::string report_;
    void* map_ = nullptr;     ///< HANDLE of the file mapping
    void* ev_req_ = nullptr;  ///< HANDLE: main -> worker
    void* ev_done_ = nullptr; ///< HANDLE: worker -> main
    void* proc_ = nullptr;    ///< HANDLE of the worker process
    void* job_ = nullptr;     ///< HANDLE of the job that kills the worker with the engine
    uint8_t* base_ = nullptr;
    int32_t* ids_ = nullptr;
    float* w_ = nullptr;
    float* x_ = nullptr;
    float* out_ = nullptr;
    int64_t req_ = 0;
};

/// `ds41_generate --gpu2-worker <mapping name> <parent pid>`: the worker's whole life.  Returns the exit code.
int ds41_gpu2_worker_main(const char* name, unsigned long parent_pid);

}  // namespace strata::v41
