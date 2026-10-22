# DeepSeek-V4.1 on Windows (native)

The DeepSeek-V4.1 engine (`tools/v41/engine`, `ds41_generate`) was Linux-only. These changes make it build and run
natively on Windows 10/11 with an NVIDIA card. There is no WSL step.

## Second merge with upstream main (2026-10-22)

Upstream `d4f6506` (109 commits) added batch slots, the conversation pool (`--conversation-cache-mib`, default 8192,
pinned), a multi-GPU layer split, a remote expert card (`--remote-device`), the GPU image encoder and `--vision-weights`,
forced tool calls in DSML, exact partial resumes and its own Windows build fixes. Where upstream does the same job,
the merge takes upstream's code:

- Image prompts are reused through the pool's position identities (upstream), not our embedding-row hashes.
- Forced `tool_choice`: upstream's `ds41_forced_call`, plus our calls-block opener (`<｜DSML｜ calls>` before the
  invoke). A bare invoke without it is what made the model fall back to text before.
- Kept from this branch: `--gpu2` (now started from a `before_tier` hook in `ds41_build_pipeline`, so it runs between
  the dense half and the tier; it cannot be combined with `--layer-split`), the blob mirror, the `dio::` layer, the
  4 GiB pinned-arena pieces (upstream's `STRATA_ARENA_PIN_AFTER` stays Linux-only), and the 16-token rewind
  (`restore_to`). The pool covers a re-rendered last token only when that turn was saved to the pool, so the rewind
  runs first for the live conversation.
- `Ds41MoeImpl` has upstream's `remote` (the in-process remote card) and our `gpu2` (the worker process) side by
  side. The in-process card cannot be used here: a second card in the engine's process loses its VRAM under WDDM once
  the 96 GiB arena is pinned (see "Dual GPU").
- Windows fixes needed in upstream's new code: `ds41_vision.py`'s new weight readers used `os.pread` and `os.open`
  without `O_BINARY`; the sidecar reader's new `close()` assumed a descriptor ours does not have, so the encoder did not
  start.

Same-day bench (fresh server per run, the pinned corpus from the first merge, cold 8K / 16K / warm 8K, 1,500 tokens
out, `--gpu2 1`, mirror on). The pre-merge exe ran under the new server code:

| | prefill (tok/s) | decode (tok/s) |
| --- | ---: | ---: |
| pre-merge exe, no vision | 507 / 630 / 437 | 24.5 / 24.0 / 23.7 |
| merged, no vision | 510 / 533 / 512 | 28.2 / 31.0 / 30.5 |
| pre-merge exe, CPU encoder | 507 / 620 / 479 | 24.0 / 23.8 / 22.4 |
| merged, CPU encoder (2 runs) | 509 / 595 / 504, 498 / 602 / 445 | 28.7-30.4, 28.8-30.0 |

8K prefill is unchanged. The 16K and warm-8K cases move by 10-15% between runs of the same exe, in both directions.
Decode is 15-25% faster after the merge. The first-merge numbers (515 / 672 / 545) were taken without the image encoder
on a different day. The pre-merge exe gives 504-507 / 553-630 today, so that gap is the machine and not the merge.

### Image encoder: CPU vs GPU

One fresh server per placement. Text bench as above, then `check_api.py`, then DeepSeek's `carrots.jpeg` (444 image
tokens) twice. The second request hits the server's image cache and reuses 454 tokens.

| encoder | prefill 8K / 16K / warm 8K | check_api (s per request) | carrots: 1st request, decode | 2nd |
| --- | ---: | ---: | ---: | ---: |
| CPU, f32, 8 threads (config) | 498 / 602 / 445 | 8.3 / 8.2 / 4.9 / 1.0, PASS | 19.9 s, 17.8 tok/s | 1.5 s |
| 5090, resident (`"device": "cuda"`) | 504 / 661 / 525 | 8.5 / 7.0 / 12.1 / 1.9, PASS | 28.3 s, **2.1 tok/s** | 16.6 s, 2.2 tok/s |
| 5090, `"offload": true` | 502 / 651 / 517 | 6.6 / 5.8 / 5.1 / 1.0, PASS | **11.7 s**, 18.6 tok/s | 1.6 s |
| 3090 (`"cuda_device": 1`, `--gpu2-reserve 7`) | 489 / 642 / 509 | 4.3 / 4.5 / 5.3 / 1.1, PASS | hung (300 s), then 400 | - |

- Resident on the 5090, the encoder holds 1.85 GiB of weights plus a 2.65 GiB pool. Every reply in an image chat then
  decodes at about 2 tok/s, while text-only requests are unaffected. The card is at 31.6 of 32.6 GB.
- With `offload`, the weights stay in pinned RAM and are copied to the card for each image. This was the fastest
  placement, with no decode cost. CUDA PyTorch (`torch 2.14.1+cu130`) is now in `venv-vision`.
- 3090: the small `check_api` pictures were the fastest, but on the photo the 3090 fell off the bus
  (`nvidia-smi`: "GPU is lost. Reboot the system to recover this GPU"). The OCuLink card is not safe for the encoder
  next to the gpu2 worker.

The config keeps the CPU encoder.

### Long chat with a side call

The case pi hits is a long chat with a short unrelated request between turns (a title or summary call). Before the
merge, the single park held one conversation. Now the pool keeps it. Test: a 113K-token document, thinking on, 4
turns, with a 17-token side call after turn 1. Run on the 5090 alone (the 3090 was lost, see above):

| request | prompt | reused | prompt read |
| --- | ---: | ---: | ---: |
| turn 1 | 113,157 | 0 | 144.4 s |
| side call | 17 | 0 | 1.2 s (the chat went to the pool) |
| turn 2 | 113,350 | 113,156 (pool) | 5.3 s |
| turn 3 | 113,485 | 113,349 (rewind) | 4.7 s |
| turn 4 | 113,608 | 113,484 (rewind) | 6.0 s |

One fix was needed. The pool's choice compared the pool against what the live state offers by upstream's rules,
which do not include the rewind. So turn 3 resumed from turn 1's pool record (112,636 of 112,771), and every later
turn would have re-read everything since turn 1. The live state now also offers `lcp` when
`Ds41Dense::can_restore_to(lcp)` says the rewind can reach it.

### Checks after the merge

`test_serve_ds41.py` (all passed), `serve/test_server.py` (318 tests, OK), `check_api.py` PASS, forced `tool_choice`
4/4 parsed calls, keep-alive: longest gap between events 20.1 s on OpenAI and Anthropic (6.5 KB tool call held
58-68 s).

## Merged with upstream main (2026-10-20)

Upstream (`Indras-Mirror/Xolotl` up to `9747b9e`) made its own native Windows port in the meantime
(`tools/v41/engine/ds41_sys.hpp`, `build-win.bat`, `docs/WINDOWS.md`, the same `GGML_CUDA_USE_GRAPHS` fix in
`CMakeLists.txt`). After the merge, `dio::` in `ds41_io.hpp` forwards the basic calls to `ds41_sys::` and only adds what
the tuning below needs: `direct_align`, `disk_space`, `mem_available_gib` and `heap_check`. One upstream fix was needed:
the new Engram hot-set loader (`Ds41Engram`, `XENGHOT1` packs) called `::pread` / `::fstat` directly and did not
compile with MSVC. It now goes through `ds41_sys::`.

The same bench as in "Second SSD" (a fresh server, then cold 8K, 16K and warm 8K requests, 1,500 tokens out,
temperature 0, thinking off, the config with `--gpu2 1` and the mirror). The prompts were cut from the pre-merge files, so
the text is the same as in the earlier runs. Two runs:

| | prefill 8K cold / 16K / 8K warm (tok/s) | decode (tok/s) | ms per verify pass | tokens per pass |
| --- | ---: | ---: | ---: | ---: |
| before the merge ("Second SSD", mirror on, 2 runs) | 498-504 / 641-647 / 515-522 | 26.0-28.2 | 94-100 | |
| after the merge, run 1 | 515 / 672 / 545 | 24.6 / 26.5 / 25.0 | 100 / 102 / 94 | 2.45 / 2.71 / 2.34 |
| after the merge, run 2 | 517 / 675 / 537 | 25.7 / 27.0 / 24.7 | 95 / 104 / 91 | 2.45 / 2.81 / 2.25 |

Prefill is 3-5% faster. Upstream's new defaults are the likely cause: a 16-row verify or prompt-tail pass is now one
tier call (`kMaxMulti` 8 -> 16), and attention keys are gathered in F16 (`STRATA_ATTN_F16_GATHER`). The time per decode
pass is unchanged. Decode tok/s moves with how many drafts DSpark accepts, so compare ms per pass. The before-merge
runs also had `MIMO_CHUNK_PROF=1` set; these did not. Not checked: whether the answers are the same text as before the
merge (upstream also changed the 1017 chat encoder), the HIP build, and Linux.

## What changed

| Linux-only piece | Windows replacement |
| --- | --- |
| `open` / `pread` / `pwrite` / `fstat` / `fsync` on the model, the mirror and the Engram tables | `tools/v41/engine/ds41_io.hpp`: Win32 handles opened `FILE_FLAG_OVERLAPPED`, so the reader threads issue positioned reads in parallel on one handle. On Linux it calls exactly what the code called before. |
| `O_DIRECT` (expert reads that skip the page cache) | `FILE_FLAG_NO_BUFFERING`. The callers already align offset, length and buffer to 4 KiB. On Windows the model file must also have no live mapping and no open buffered handle while these reads run (see "SSD reads on Windows" below). |
| `posix_fadvise` | No Windows counterpart, so it does nothing there. |
| `posix_memalign` / `std::aligned_alloc` + `free` | `_aligned_malloc` / `_aligned_free` (`dio::aligned_alloc` / `dio::aligned_free`) |
| `/proc/meminfo` MemAvailable (the arena's RAM check) | `GlobalMemoryStatusEx` |
| Engram raw block-device reads (FIEMAP + `/sys/dev/block`) | Off on Windows. The tables are read through the filesystem. |
| `--engram dir1:dir2` | `--engram dir1;dir2` on Windows (drive letters contain a `:`) |
| `setenv`, `strcasecmp`, `<strings.h>` | `_putenv_s`, `_stricmp` |
| `src/kernels/cpu/ptq1_0.cpp` GCC `__attribute__((target))`, `__builtin_cpu_supports`, `_cvtsh_ss` | Empty on MSVC (MSVC compiles AVX2 intrinsics without a target). FMA/F16C are covered by `cpu_avx2_ok()`. |
| `ds41-engine.sh` (flock GPU lock, `systemd-run` 84 GiB RAM cap, no swap, the env defaults) | The env defaults are now in `tools/v41/serve/strata-ds41-windows.json` (`"env"`). There is no RAM cap or GPU lock on Windows. |
| The `/tmp` text log in `serve_ds41.py` | The OS temp folder (Linux still gets `/tmp`) |

### A bug that only Windows showed: two layouts of `ggml_backend_cuda_context`

`strata_mmq` (and `strata_mmq_w4a4` / `strata_kq`) include ggml-cuda's `common.cuh`, and `moe_mmq.cu` constructs
`ggml_backend_cuda_context` itself. With `STRATA_GGML_CUDA=ON`, ggml-cuda is compiled with `GGML_CUDA_USE_GRAPHS`, and
that define adds members to the struct (`cuda_graphs`, an `unordered_map`, and more). `strata_mmq` was compiled without
it. The two inline copies of the constructor therefore disagreed on the layout. MSVC's linker kept the larger one, which
built the map past the end of the smaller allocation. The first prompt then died with heap corruption (0xC0000374)
inside the CUDA driver's next allocation. GNU ld happens to keep the matching copy, which is why Linux never showed it.
The fix (`tools/ds4/cmake/fork.cmake`, at the end) gives those targets the same `GGML_CUDA_USE_GRAPHS`.

How it was found: `-DSTRATA_DS41_DEBUGINFO=ON` (symbols) plus the engine's Windows crash handler, which prints the stack
to the log. Then `DS41_HEAP_CHECK=1`, which runs `HeapValidate` on every process heap at checkpoints, narrowed it to the
MMQ setup in `gpu_run_chunk`. AddressSanitizer (`-DSTRATA_DS41_ASAN=ON`) did not catch it, because its allocator pads
every block.

## Build

You need Visual Studio 2022 (or 2019) Build Tools with C++, CUDA 13.x, `cmake` + `ninja`, and git.

```bat
git -c http.sslBackend=schannel -c core.autocrlf=false clone https://github.com/ggml-org/llama.cpp.git third_party\llama.cpp
cd third_party\llama.cpp && git checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
rem the patch must be applied with LF line endings (a CRLF checkout of this repo gives a CRLF patch file)
python -c "import sys;d=open(sys.argv[1],'rb').read().replace(b'\r\n',b'\n');open(sys.argv[2],'wb').write(d)" ..\..\tools\xolotl\ggml\xolotl-ggml.patch %TEMP%\xolotl-ggml.patch
git apply %TEMP%\xolotl-ggml.patch
cd ..\..
build-ds41-windows.bat
```

`CUDA_ARCH` picks the card. By default the script builds for every NVIDIA card `nvidia-smi` lists (RTX 50 = `120a`,
RTX 40 = `89`, RTX 30 = `86`; `120a` if it finds none), so on the RTX 5090 + RTX 3090 PC it builds `120a;86`, which
the second card (`--gpu2`, see "Dual GPU" below) needs. RTX 50 needs the arch-specific `120a` rather than `120`,
because ggml's MXFP4 MMQ uses block-scaled MMA. `set CUDA_ARCH=120a` before the script builds for one card only. A
build made for other architectures is reconfigured (its CUDA objects compile once more).

## Run

Edit the model paths in `tools/v41/serve/strata-ds41-windows.json`, then:

```bat
START-DS41.bat
```

The API is at `http://127.0.0.1:8150/v1` (OpenAI and Anthropic).

The config reads half of each SSD expert from a second SSD (`STRATA_BLOB_MIRROR`, see "Second SSD"). Make that file
once with `MAKE-DS41-MIRROR.bat` (85 GiB on `C:`); without it the engine logs `blob mirror: NOT used` and runs as before.

## Measured (2026-10-19)

Setup: RTX 5090 32 GB, Ryzen 7 7800X3D, 159 GB RAM, Windows 11, NVMe (NTFS), driver 610.88, CUDA 13.1. Model: GSQ
1017 (163 GiB), 64 GiB pinned arena, the config's defaults.

| | |
| --- | --- |
| Load (one-shot, ctx 8192) | 55 s; the arena pins at 4 KB pages |
| Server start (512K context) | ~80 s to `ready` |
| One-shot greedy decode, no DSpark (22-token prompt, 103 tokens) | 17.2 tok/s |
| Server, DSpark on, warm (24-token prompt, 460 tokens) | 16.8 tok/s, 3.03 tokens per verify pass |

Output checked by hand: correct Python for "is prime" and "word frequency" prompts.

Not measured: the 512K context filled, tool calls, the blob mirror (since measured: "Second SSD"), and Windows' `FILE_FLAG_NO_BUFFERING` +
overlapped reads against Linux `O_DIRECT` on the same disk. Prefill: see the two sections below.

### VRAM (the same 700-token request, temperature 0, one run each, server restarted between runs)

During decode the card is full: 31.7-31.9 of 32.6 GB in use, of which the engine holds 29.6 GiB and the rest is the
desktop and other apps on the same card. The engine's "shared" GPU memory (64.2-64.8 GiB) is the pinned RAM arena and
the pinned staging buffers, not VRAM spilling into system RAM: its dedicated VRAM stays flat while it decodes.

| `--vram-margin` | expert slots | ms per DSpark verify pass | tok/s |
| ---: | ---: | ---: | ---: |
| 0.5 | 22.62 GiB | 343 | 8.3 |
| 1.0 (the Linux default) | 22.12 GiB | 151 | 20.1 |
| **1.5 (this config)** | 21.62 GiB | 145 | 23.4 |
| 2.0 | 21.12 GiB | 146 | 21.0 |

tok/s also moves with how many drafts DSpark accepts (2.85-3.38 tokens a pass here), so ms per pass is the fairer
column. With under ~0.7 GB left on the card (margin 0.5), Windows' display driver competes for memory and every pass
gets slower. 1.5 sits on the plateau with room for the desktop, so the Windows config uses it. The bigger win is VRAM
that other programs hold, about 2 GB here: about 115 more experts fit per GiB freed. Plugging the monitor into the
motherboard (the CPU's graphics) or closing GPU-heavy apps frees it.

### Long prompts, long answers, cold and warm (`tools/v41/bench_long.py`, server, `--vram-margin 1.5`)

Each prompt is new text (no shared prefix), temperature 0, 1500 tokens out, thinking off. "Cold" = the first request
after the server started (empty VRAM expert cache). Measured before the SSD read fix below; the table has not been
re-run since. One check after the fix (fresh server, DSpark on, an 8,054-token prompt of the same kind but other
text, 200 tokens out): prefill 30.3 s = 266 tok/s (dense 9.9 s, experts 17.7 s), decode 16.5 tok/s.

| case | prompt tokens | prefill | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| cold 2K | 2,055 | 30.4 s | 68 | 15.5 |
| 8K | 8,054 | 37.0 s | 218 | 17.6 |
| 16K | 16,049 | 59.5 s | 270 | 16.1 |
| 32K | 32,079 | 95.7 s | 335 | 17.8 |
| warm 2K | 2,055 | 26.1 s | 79 | 16.0 |
| warm 8K | 8,054 | 36.1 s | 223 | 18.9 |
| 32K conversation + a new question, after two unrelated requests | 33,605 (0 reused) | 85.4 s | 394 | 16.0 |

Follow-up turns sent right after their turn (16K conversation, 400 tokens out): the prefix is reused.

| turn | prompt tokens | reused | prefill | decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16,068 | 0 | 57.0 s | 16.6 |
| 2 | 16,481 | 16,467 | 3.0 s | 15.3 |
| 3 | 16,894 | 16,880 | 2.5 s | 18.1 |

### Where the time goes

**Prefill:** a prompt batch (up to 12,288 tokens) streams every routed expert through the card once: the RAM arena
over PCIe, and the file-tier experts (77.6 GB for one batch here) from the SSD. The file-tier reads set the speed:
before the fix below they ran at 3.8 GB/s and were 34 of a 16K prompt's 58 s. They now run at 7.0 GB/s, the drive's
rated read speed:

| one-shot, `--prompt-batch 12288`, the config's args, `-n 16` | before | after |
| --- | ---: | ---: |
| 8K prompt (8,050 tokens, 1 batch), `MIMO_CHUNK_PROF=1` | 35.94 s = 224.0 tok/s | 23.24 s = 346.3 tok/s |
| 16K prompt (16,045 tokens, 2 batches), two runs each | 58.23 / 58.34 s = 275 tok/s | 37.14 / 37.20 s = 431 tok/s |
| file-tier read rate (`MIMO_CHUNK_PROF=1`, 8K / 16K) | 3.74 / 3.79 GB/s | 6.97 / 6.96 GB/s |
| 16K: the expert phase / of it waiting on the SSD | 41.5-41.9 s / 33.9-34.2 s | 27.7-27.8 s / 18.4 s |
| 16K: the dense half | 15.6-15.8 s | 8.8-9.0 s (one profiled run 15.0) |
| arena seed at load (64 GiB) | 16.1-16.2 s | 14.0-14.3 s |

The last-row logits (`--dump-last`) and the generated ids are byte-identical between before and after in every run.
The prompts are this repository's files cut to length, as `tools/v41/bench_long.py` builds them (raw ids, no chat
template). The dense half's gain came with the embedding copy (the first change below); it was not separated further.

### SSD reads on Windows (`tools/v41/engine/ds41_rdbench.cpp`)

The benchmark replays the engine's read pattern outside the engine: the file-tier blobs at their real offsets in the
GGUF (`ds41_rdbench_spans.py` lists them from `<model>.gguf.experts.json`, a random half of each layer's experts, mean
blob 10.2 MiB), N threads, the same `dio::pread`. Samsung 990 PRO 4TB (PCIe 4.0 x4, NTFS, 512-byte logical sectors):

| pattern | GB/s |
| --- | ---: |
| unbuffered, one read per span (3 a blob), 4 / 8 / 16 / 32 / 64 threads | 5.12 / 6.93 / 6.96 / 7.08 / 7.09 |
| unbuffered, the prompt's wave reader (32 threads, each blob in 8 parts) | 6.88-7.08 |
| the same with an 8 GiB pinned buffer copied to the card alongside (bounce + memcpy / read in place) | 4.76-5.07 / 6.89-7.04 |
| the same while the process holds a mapping of the model file (only ~10 MB of it touched) | 3.75 |
| the same right after 8 GiB were read through a buffered handle of the file, handle kept open | 3.45 |
| ... after that handle is closed: at once / 5 s later | 2.47-2.74 / 6.86 |
| with a mapping or an open buffered handle, larger requests (2 parts, 1 part, 4 MiB) | 5.15-5.40 |
| wave reader with `FILE_FLAG_SEQUENTIAL_SCAN` / `FILE_FLAG_RANDOM_ACCESS` / neither | 6.88-6.91 / 6.90 / 6.90-6.91 |
| every request of a blob in flight at once (overlapped, 512 KiB-1 MiB pieces, 8-16 threads) | 6.87 |

So the drive gives 7.0 GB/s to this pattern from 8 threads up, with one blocking read per thread. More threads,
other request sizes, more requests in flight per thread and the access-pattern flags all measured the same, so an IOCP
queue in `ds41_io.hpp` (not built) has nothing left to win. The engine already had the unbuffered handle open and in
use (0 buffered fallbacks counted). What cost half the rate: while a file has a live
mapping or an open buffered handle (with cached data, or opened right after a mapping), NTFS checks cache coherency
on every unbuffered read of it. The cost is per request, which is why fewer, larger requests recovered part of it, and
it ends a few seconds after the last mapping or buffered handle goes. The engine had all three: `Ds41Dense` kept
`GgufModel`'s mapping of the 163 GiB file for the token embeddings, the RAM arena was seeded through the buffered fd,
and `Ds41FileBlobs` kept that buffered fd open. Inside the engine, a read probe after `Ds41FileBlobs::open` ran at
3.4 GB/s before the fix and 7.0 after it. The changes are Windows-only (`#if defined(_WIN32)`); Linux runs the code it
ran before:

- `Ds41Dense` (GPU backend) copies `token_embd` (F16, ~1 GB) to RAM and drops its mapping after the weights are
  uploaded. DSpark takes that copy (`Ds41DSparkConfig::token_embd`) and drops its own mapping of the trunk after load.
  This matters for the server, where DSpark is on (one server check above; the full server table not re-run).
- `Ds41FileBlobs` opens its buffered fd only on first use (`buffered_fd()`, for an unaligned fallback read or the
  mirror's size check). `read()` (the arena seed) goes through the unbuffered `read_part`.
- `read_part` reads a piece straight into its destination when the offset, length and address are multiples of the
  volume's logical sector (`dio::direct_align`, 512 here), and skips the 4 KiB bounce and its memcpy: 8K file tier
  6.64 -> 6.97 GB/s, prompt 23.83 -> 23.24 s. `STRATA_READ_INPLACE=0` turns it off.

`STRATA_WAVE_PARTS=2` instead of 8 measured the same after the fix (6.99 GB/s, 23.13 s), so the default stays. Found
on the way, not changed here: the Engram tables cost the 8K prompt's dense half 3.6 s (9.0 s with them, 5.4 s
without, before the fix), and for a few seconds after an engine exits (or a build writes to the same drive) the
reads run at 1.7-2.7 GB/s.

**Decode** (one-shot profile, `STRATA_DEC_PROF=1 STRATA_ENG_PROF=1 DS41_PROF_X=1`, 700 tokens, DSpark at 3.07
tokens a pass): one verify pass = 138 ms, made of the dense half 46 ms (attention, shared expert, head on the GPU) and
the expert tier 102 ms. Per pass the tier serves ~768 experts from VRAM, ~65 from the RAM arena over PCIe (~0.6 GB)
and ~9 from the SSD, and waits ~46 ms for those SSD reads. So the SSD serves about 1% of the experts but costs about
a third of each pass. The Engram tables cost 3.5 ms per decode read call. The CPU tier was almost unused (3 experts in
700 tokens).

The GPU link on this PC runs at PCIe 4.0 x16 (the card supports 5.0), which halves the arena's DMA rate.

### RAM arena size (`--arena-gib`, measured 2026-10-20)

The arena holds the routed experts that are not in VRAM; the rest are read from the SSD. More arena means fewer SSD
reads. On this PC (159 GB RAM, the setup above) the config now uses **96 GiB** (it was 64).

Each run: a fresh server through `START-DS41.bat` with a copy of the config that changed only `--arena-gib` (plus
`STRATA_ENG_PROF=1`), then three requests in a row, each new text (the files `tools/v41/bench_long.py` uses),
temperature 0, thinking off, 1500 tokens out: a cold 8K prompt (the first request, empty VRAM cache), a 16K prompt,
then a warm 8K prompt. "ms/pass" is decode time per DSpark verify pass, which does not move with how many drafts
are accepted (2.4-2.9 tokens a pass here), so it compares sizes more fairly than tok/s. "Free RAM" is the lowest
`\Memory\Available MBytes` while the requests ran.

| arena | load | RAM / SSD experts | prefill 8K cold / 16K / 8K warm (tok/s) | decode cold / 16K / warm (tok/s) | ms/pass | decode VRAM hits | free RAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 GiB (second run) | 56 s | 6,774 / 8,586 | 311 / 408 / 323 | 20.5 / 19.4 / 19.5 | 118 / 134 / 121 | 85-88% | 66.6 GiB |
| 80 GiB | 59 s | 8,303 / 7,057 | 345 / 458 / 361 | 22.3 / 20.0 / 21.4 | 110 / 132 / 119 | 85-88% | 47.2 GiB |
| **96 GiB** | 62 / 64 s | 9,796 / 5,564 | 376 / 500 / 394, 372 / 497 / 391 | 24.7 / 22.4 / 23.2, 24.9 / 22.3 / 22.3 | 108-121 | 85-88% | 34.6-35.9 GiB |
| 100 GiB | 64 / 64 s | 10,171 / 5,189 | 384 / 515 / 404, 388 / 521 / 407 | 22.8 / 22.1 / 23.5, 23.0 / 21.7 / 21.3 | 109-118 | 84-87% | 30.8 GiB |
| 104 GiB | 65 s | 10,539 / 4,821 | 399 / 498 / 421 | 20.8 / 20.1 / 20.3 | 113 / 131 / 124 | 80-84% | 27.0 GiB |
| 108 GiB | 66 s | 10,908 / 4,452 | 396 / 423 / 316 | 19.7 / 18.3 / 18.5 | 134 / 136 / 129 | 76-79% | 20.7 GiB |
| 120 GiB | 70 s | 12,026 / 3,334 | 124 / 127 / 113 | 16.3 / 15.7 / 16.7 | 147 / 172 / 172 | 65-68% | 9.8 GiB |

The first 64 GiB run came right after a reboot (the Engram tables not yet in the file cache) and measured 292 / 358 /
298 tok/s prefill, so the table uses the second run, which came after the others. The arena seed takes 14 s at 64 GiB
and 27.5 s at 120 GiB.

Up to 96-100 GiB, each extra GiB moves experts off the SSD: 64 -> 96 GiB is about 20% more prefill and 5-10% less
time per decode pass. 100 GiB measured 3-4% more prefill than 96 and the same decode (two runs each). Past that,
the card loses VRAM. On Windows the driver lets the card use only as much VRAM as it could also back with free system
RAM, and pinned memory is not free. In a stand-alone test (`cudaMemGetInfo` while registering host memory in 4 GiB
pieces), free VRAM stayed at 30.26 GiB up to 96 GiB registered and was 18.94 GiB at 112 GiB. In the server this
shows as `serve: VRAM 17.0 GiB free -> prompt batch 12078` at 104 GiB, 13.0 GiB (batch 7783) at 108 and 6.1 GiB
(batch 512) at 120: smaller prompt batches, a smaller VRAM expert cache after the prompt (fewer VRAM hits), and
slower decode. At 120 GiB Windows also started paging (up to 179,000 pages/s out, the pagefile grew by 2.2 GB).

96 GiB is the config's value rather than 100: the two measured within a few percent, the slowdown starts between
100 and 104 GiB, and it depends on free RAM, so 96 leaves about 4 GiB more for other programs before it begins. For
another PC: keep about 35 GiB of RAM free beyond the arena and the engine (`\Memory\Available MBytes` during a
request); below about 30 GiB the card starts losing VRAM.

The Engram tables are read through the file cache, which a large arena shrinks. Their decode reads (`engram decode
reads`, per request) measured 1.2-1.4 ms a call at 64 GiB (second run), 0.8-1.6 at 96 and 100, and 1.5-1.8 at 108
and 120, so they did not set the limit.

**A pinning limit found on the way.** After a reboot, one `cudaHostRegister` of the 64 GiB arena failed (out of
memory), so the arena was not pinned (`arena: ... cudaHostRegister FAILED (out of memory) - the arena is NOT pinned`).
The day before, the same 64 GiB registered whole. In the stand-alone test one call of 63 GiB was taken and 63.9 GiB
was refused, while 100 GiB in 4 GiB pieces registered and copied to the card at 28.8 GB/s, so the limit is per call,
not a total. `build_arena` (`ds41_moe.cpp`) now gives `PinnedArena` pieces about 4 GiB apart as a fallback:
`PinnedArena` still registers the whole arena first and uses the pieces only when that is refused. A copy that
crosses two registrations fails (invalid argument), so each piece starts at an expert slot, page-aligned, and every
copy out of the arena moves one expert from inside its slot. The log then says `cudaHostRegister of the whole arena
FAILED (out of memory); 24 slices pinned (95 GiB)` (95 = 96 GiB rounded down). Output was checked by reading the
answers' openings at every size. The Linux path is unchanged where a whole registration succeeds; the HIP build was
not run.

Not measured: sizes between 100 and 104 GiB, more than one or two runs per size, prompts above 16K, the 512K
context filled, other programs using RAM during the run, and a PC with a different amount of RAM.

## Dual GPU (`--gpu2`, measured 2026-10-20)

An RTX 3090 (24 GB, sm_86) on OCuLink next to the RTX 5090 now holds a second VRAM expert cache. The config uses it
(`"--gpu2", "1"`). Prompts read 19-22% faster and a decode pass takes 12-13% less time than on the 5090 alone:

| server, arena 96, margin 1.5 | prefill 8K cold / 16K / 8K warm (tok/s) | decode (tok/s) | ms per verify pass | SSD experts per prompt |
| --- | ---: | ---: | ---: | ---: |
| 5090 alone, old build (`120a`) | 378 / 506 / 399 | 24.7 / 22.3 / 22.5 | 112 / 120 / 111 | 4,237 / 6,522 / 4,109 |
| 5090 alone, this build (`120a;86`, no `--gpu2`) | 377 / 503 / 397 | 24.6 / 21.6 / 22.3 | 113 / 117 / 112 | 4,237 / 6,534 / 4,127 |
| **5090 + 3090 (`--gpu2 1`, the config)** | **459 / 601 / 480** | **27.0 / 25.4 / 26.3** | **100 / 104 / 97** | 2,648 / 3,858 / 2,451 |

The bench is the one in "RAM arena size": a fresh server, then three requests in a row (cold 8K, 16K, warm 8K; new
text each; 1,500 tokens out; temperature 0; thinking off). "ms per verify pass" is the fairer decode column because
tok/s also moves with how many drafts DSpark accepts (2.4-2.8 tokens a pass in every run here). The three requests
took 271 s on one card and 233 s on two. The answers are coherent and on topic, but they are not the same text as on
one card. With `--route-bias`, the experts on the 3090 count as VRAM-resident, which shifts the routing. The sums of
the two cards are also added in a different order. With `--gpu2` off, the build gives the old numbers (the first two
rows).

### The link

Measured with the CUDA runtime (`cudaMemcpyAsync`, 256 MiB, pinned memory registered portable + mapped like the
arena):

| | RTX 5090 | RTX 3090 |
| --- | --- | --- |
| link (`nvidia-smi -q`) | PCIe 4.0 x16 | PCIe 4.0 x4 (OCuLink; it shows gen 1 when idle) |
| host -> card / card -> host | 28.6 / 28.4 GB/s | 6.70 / 6.57 GB/s |
| both cards copying at once | 28.5 GB/s | 6.7 GB/s (no contention) |
| peer access (`cudaDeviceCanAccessPeer`) | no | no (WDDM; `nvidia-smi topo` says PXB) |
| `cudaMemcpyPeer` 5090 <-> 3090 | 6.5 / 6.4 GB/s (the driver goes through host RAM) | |
| one small copy (4 B - 28 KB) | 8-11 us | 8-12 us |
| 256 KB / 1 MiB | 17-20 / 44-48 us | 46 / 164-170 us |

A round trip 5090 -> host -> 3090 -> host -> 5090 chained with events took 39 us for 28 KB (one token's activations),
61 us for 112 KB, and 37 ms for 96 MiB (a 12K-token prompt batch in bf16). Small transfers cost latency, not
bandwidth, so a split that crosses the link once per layer costs ~40 layers x ~40 us = ~2 ms of a ~110 ms decode pass
if nothing overlaps it.

### Why the 3090 runs in a second process (WDDM)

On Windows the driver backs card memory with system RAM, and pinned memory counts against that. Two cards in ONE
process cannot keep both their VRAM and the 96 GiB arena. Measured: host RAM pinned in 4 GiB `cudaHostRegister`
pieces, then each card filled to its free VRAM and read back by a kernel. Memory that WDDM quietly put in system RAM
reads at PCIe speed, so the read rate shows where the memory really is.

| pinned host RAM | one process, both cards | one process, 5090 only | two processes (5090 + arena / 3090 alone) |
| ---: | --- | --- | --- |
| 48 / 60 GiB | 29.5 + 22.0 GiB, 1,340-1,385 / 855 GB/s | | |
| 64 GiB | 5090 6.5 GiB, 3090 0 GiB | | |
| 80-96 GiB | registering past 64 GiB fails | 29.5 GiB, 1,382 GB/s at 96 | 29.5 + 22.0 GiB, 1,386-1,427 / 857 GB/s at 96 (and at 80, 88) |

So the 3090 runs in a helper process: `ds41_generate --gpu2-worker`, which the engine starts itself. With 96 GiB
pinned in the engine and the 3090's 22 GiB in the helper, free RAM did not drop by the 3090's VRAM. The two processes
talk through a pagefile-backed shared memory section and two events (`tools/v41/engine/ds41_gpu2.{hpp,cpp}`). The
engine registers the section's result rows with CUDA (`cudaHostRegister` of a `MapViewOfFile` view works, mapped
pointer included), so its add kernel reads the 3090's sums in place.

Two things to know when measuring this. If a card has idled for ~10 s, the first read-back runs at 60-200 GB/s (P8
clocks), so warm it up first. And a pinning test that takes a card's VRAM away can leave the 3090 driver logging an
error (`nvlddmkm` event 153); the card worked again in the next process.

### What could be split, estimated before building

From "Where the time goes": a decode pass is ~1/3 dense half (attention, shared expert, head) and ~2/3 expert tier
(VRAM hits, arena experts over PCIe, ~1/3 of the pass waiting on SSD reads). A prompt batch streams every routed expert
once, and its file-tier SSD reads set its speed.

- **(a) Expert split (built, chosen).** The 3090 holds ~20 GiB of experts, computes the routed rows for them, and
  only the activations and its sums cross the link. Those experts leave the 5090's RAM arena, so the arena holds
  ~2,000 more experts and the SSD serves fewer: one 12K-token batch reads ~1,600 fewer blobs, ~15 GB, ~2 s at
  7 GB/s. The 3090 keeps its cache during prompts, while the 5090 gives its own back to the prompt buffers. Cost per
  layer call: one round trip, which overlaps the 5090's own share of the experts.
- **(b) Layer split of the dense half.** A decode pass runs one request, so moving layers to the 3090 puts them in
  series, not in parallel. The dense half takes 46 ms a pass on the 5090, and the 3090 has about half its memory
  bandwidth (936 vs 1,792 GB/s), so half the layers there adds ~20 ms to a ~110 ms pass. In return it frees only those
  layers' dense weights on the 5090, and the 3090's arena misses would cross its x4 link (6.7 GB/s instead of 28.6).
  In one process it also caps the arena at ~60 GiB (above). Not built.
- **(c) DSpark drafter, or long-context KV / attention, on the 3090.** The drafter frees 2.9 GB on the 5090 (~16%
  more expert slots there), but it runs in series with every verify pass, needs that pass's hidden states sent
  across, and would draft on the slower card. KV / attention sits inside each layer's dense half: two crossings a
  layer is ~3 ms a pass, for VRAM that only matters once the context fills the 5090. Neither looked worth more than a
  few percent. Not built.
- **(d) The 3090 as a staging / decompress helper.** Without peer access, anything the 3090 prepares goes back
  through host RAM and then over the 5090's own link: the same bytes plus another hop. The experts are used in their
  packed form, so there is nothing to decompress. Not built.

Kernels on sm_86: the tier's expert kernels (native grouped MMVQ, Strata's MMQ incl. the PTQ1_0 -> Q2_0 transcode)
build and run on the 3090. MXFP4 block-scaled MMA is sm_120a-only and sits in ggml's dense half, which stays on the
5090. The worker runs one expert through the kernels before it reports ready, so a build without `86` fails at start
with a message (the engine then runs on one card).

### What the first version measured, and what fixed it

| step (same bench) | prefill (tok/s) | ms per pass |
| --- | ---: | ---: |
| expert split, first version | 387 / 532 / 419 | 102 / 109 / 108 |
| + MMQ for the 3090's resident experts | 453 (8K cold, one request) | |
| + the 3090 holds the experts after the 5090's decode cache (`--gpu2-skip auto`), no keep-warm | 429 / 580 / 450 | 109 / 120 / 108 |
| + keep-warm, `auto` | 430 / 579 / 454 | 94 / 105 / 98 |
| + keep-warm, the 3090 holds the ranking's head (`--gpu2-skip 0`, the default) | 458 / 599 / 478 | 100 / 103 / 96 |

- **Prompt: MMQ for resident experts.** The first version gave prompts only +2-5%. The engine waited ~110 ms a
  layer for the 3090, which was busy at 90-100% SM, while the SSD saving (37% fewer file-tier blobs) had already
  happened. Cause: MMQ needs a run of experts at a fixed stride, but cache slots are one blob apart, so every resident
  expert ran on the MMVQ kernel (~10x slower at prompt widths). The 5090 never hits this, because it streams its
  prompt experts into strided halves. `Ds41MoeConfig::chunk_resident_mmq` (worker only) copies the resident blobs
  device to device into those halves first. After that the 3090 takes 90-100 ms a layer, under the 5090's own share,
  so the engine no longer waits for it in prompts.
- **Decode: keep-warm.** Each layer call gives the 3090 ~0.1-0.3 ms of work, and between calls the driver drops it
  to ~700 MHz core and half its memory clock (`nvidia-smi dmon`). A call then took 1.1-1.3 ms, and the engine waited
  ~0.5 ms a layer. Between calls the worker now runs 50 us spin kernels for up to 200 ms after the last call
  (`STRATA_GPU2_WARM=0` turns this off). The card stays at 1,950 MHz / full memory clock and a call takes ~0.46 ms.
  It costs power: the 3090 draws ~170 W while decoding instead of ~100 W (prompts: 290-370 W either way). A first try
  ran the spins on their own thread and stream, alongside the tier's kernels. The worker then died with
  `unspecified launch failure` (and an `nvlddmkm` event 153 on the 3090) 4 times in the 4 server runs that used it. With the spins in
  the worker's own loop, between calls, the next 3 server runs (9 requests) had no failure.
- **Which experts go to the 3090.** `--gpu2-skip auto` leaves the ranking's head to the 5090's decode cache and gives
  the 3090 the next ~20 GiB. `0` gives the 3090 the head. Decode measured the same within noise (average 98 ms a pass
  both). Prompts were 5% faster with `0` in all three runs (456-458 / 597-599 / 478 tok/s against 429-431 / 578-580
  / 450-454), so `0` is the default.

### How it runs

- `--gpu2 N`: card N as `nvidia-smi` numbers it (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). The server still gives the
  engine `"gpu": 0`; the worker gets `CUDA_VISIBLE_DEVICES=N` for itself.
- The worker opens the GGUF itself and fills its VRAM from `--rank`, minus `--gpu2-reserve` GiB (default 3.0; its
  prompt buffers). Here that is 2,275 experts, 19.66 GiB, 3.0 GiB free, seeded in 4-7 s. The engine leaves those
  experts out of its own VRAM cache and RAM arena (`Ds41MoeConfig::exclude`). The arena then holds 9,344 experts
  instead of 9,796, but the file tier drops from 5,564 to 3,741.
- Per layer call (prompt chunk, verify pass, single token), the tier sends the rows that route to a 3090 expert:
  activations in f32, plus ids and weights with -1 / 0 for the other entries. It computes its own share meanwhile,
  with the 3090's entries as id -1 (no expert; new in `gpu_run_n_ex` / `gpu_run_chunk`). Then it adds the 3090's
  sums into the routed rows on the 5090. With nothing of a call on the 3090, the call is the one-card path unchanged.
  A 12K-row prompt sends ~200 MB each way per layer, behind the 5090's ~300 ms of its own work.
- Engine start: 62 s alone, 66-69 s with the 3090 (the server was ready after 69-71 s, twice after 95 s).
- If the worker fails, the engine finishes the request on the 5090 alone. The 3090's experts then come from the SSD,
  since they are not in the arena, so that request is slower (18.6-21 tok/s in the runs that hit it). It starts a new
  worker before the next request (measured 5.4 and 7.5 s; the next requests used the 3090 again). A missing card or
  a build without its architecture: the engine says why and runs on one card.
- Log lines: `gpu2: card 1 - ...` at start; `... + N on gpu2` on the prompt line; `, N gpu2` on the decode line;
  `serve: gpu2 so far: ...` with the time the engine waited for the 3090 after its own work, split into decode and
  prompt calls.
- Options: `--gpu2-reserve GiB`, `--gpu2-gib GiB` (cap its cache), `--gpu2-skip GiB|auto`, `--gpu2-piece rows`
  (rows per worker prompt call, default 4096). Env: `STRATA_GPU2_WARM=0`, `STRATA_GPU2_SPIN_US` (worker poll time
  without keep-warm, default 3000), `STRATA_GPU2_ENV="K=V;K=V"` (settings for the worker only, e.g.
  `MIMO_CHUNK_PROF=1;DS41_PROF_X=2` to profile it).
- Build: `build-ds41-windows.bat` now builds for every card `nvidia-smi` lists (`120a;86` here). The engine is the
  same file for both processes.
- Windows only: the helper process exists because of WDDM's pinning limit. On Linux, `--gpu2` says so and the engine
  runs on one card (not built or run on Linux / HIP here; the new code is guarded by `_WIN32` and `DS4_MOE_CUDA`).

Not measured: prompts above 16K, the 512K context filled, one-shot (non-server) runs with `--gpu2`, the 3090 on a
faster link, sessions longer than these runs (about 1.5 hours of `--gpu2` server time in all), and the cause of the
`nvlddmkm` event 153 (Xid code not decoded).

## Second SSD (blob mirror, measured 2026-10-20)

The config now reads the SSD-tier experts from two drives at once: half of each blob's parts from the model file on
`H:`, half from a copy of the coldest experts on `C:`. Prompts got 8-10% faster at 8K / 16K and 13-15% faster at 2K;
decode did not change.

### The drives

| | `H:` (the model) | `C:` (Windows, pagefile 12.8 GB) |
| --- | --- | --- |
| drive | Samsung 990 PRO 4TB | Samsung 980 PRO 2TB (heatsink) |
| link | PCIe 4.0 x4 behind the chipset (`GPP7` -> chipset, which also carries the USB drives) | PCIe 4.0 x4 on the CPU's own lanes (`GPP1`) |
| free | 507 GB | 872 GB before the mirror, ~780 GB after |

The 5090 (`GPP0`, x16) and the 3090 (`GPP8`, OCuLink x4) are on CPU lanes too, so nothing else in the engine's path
shares the chipset uplink with `H:`.

`ds41_rdbench` (the prompt's wave reader: 32 threads, blobs in 4 parts, unbuffered, in place) on 1,013 expert blobs
(12.3 GB, mean 10.2 MiB). For `C:` they were a 16 GiB slice of the GGUF copied there, at the same offsets inside the slice:

| | GB/s |
| --- | ---: |
| `H:` alone | 7.04-7.06 |
| `C:` alone | 6.60-6.70 |
| both at once, each process on its own drive (6 passes each) | 7.04-7.06 + 6.60-6.70; 148 GB in 11.3 s wall = 13.1 GB/s |

The two links do not cap each other: together the drives give about the sum.

### What the SSD still costs (before the mirror)

From the engine's `MIMO_CHUNK_PROF=1` lines (config as above, arena 96, `--gpu2 1`):

| request | file-tier read | the expert phase | waits on reads (counted) |
| --- | --- | ---: | ---: |
| 8K prompt | 29.0 GB at 6.85 GB/s (reader 5.5 s) | 9.5-9.8 s | 0.36 s |
| 2K prompt | 23.7-25.0 GB at 6.85 GB/s (reader 3.9-4.3 s) | 7.6-7.8 s | 0.24-0.30 s |
| a 310-token follow-up turn | 7.2-7.9 GB at 6.7 GB/s | 3.8-3.9 s | 0.08 s |
| decode | ~1 SSD expert per verify pass (544 in 1,500 tokens), ~1.5 ms each | | |

The counted waits suggested at most 2-3% to win. The prompt's expert phase dropped by more than that once the reads
were faster (below): the reader's other effects on the batch (likely its share of host memory bandwidth next to the
arena's DMA, not measured separately) are not in that count.

### Options, estimated first

- **(a) Mirror the coldest experts and split each blob's reads between the drives (built, chosen).** It already
  existed from Linux: `ds41_generate --build-mirror` writes the coldest experts by `--rank` into one packed file, and
  with `STRATA_BLOB_MIRROR=<file>` every split read (`read_part` with more than one part: the prompt's wave reader, the
  decode misses and the speculative stash) takes `STRATA_BLOB_MIRROR_SHARE` of each mirrored blob's parts from it. The
  file tier is the tail of `--rank` (the arena and the VRAM caches take the head), so a mirror of the coldest 85 GiB
  (8,200 experts) holds all 3,741 file-tier experts of this config, the ones `--arena-adapt` swaps out of the arena
  later, and the file tier of a 64 GiB arena without `--gpu2` (8,586 at most). Expected: the prompt's file tier from
  6.9 to ~13 GB/s.
- **(b) Move an Engram table to `C:`.** Its rows are read through the file cache, 256-byte rows in 4 KiB pages, during
  the dense half; the expert reads are in the expert phase. The two hardly overlap, and the dense half did not change
  with the mirror (5.2-5.8 s for 8K either way). 95 GB each. Not built.
- **(c) A full copy of the GGUF (163 GiB) and per-read drive choice by queue depth.** The share split already keeps
  both drives busy on every blob, and the hot experts never come from the SSD. Twice the space of (a) for experts that
  live in RAM or VRAM. Not built.

### Windows changes to the mirror

- `open_mirror` checked the model file's size through `buffered_fd()`, which on Windows opens a buffered handle of the
  model file. While one is open, NTFS checks cache coherency on every unbuffered read (7.0 -> 3.4 GB/s, "SSD reads on
  Windows"). It now uses the unbuffered handle.
- Mirror parts are read straight into their destination when sector-aligned (as `read_part` already did on the model
  file), instead of through the bounce and a memcpy.
- `--build-mirror` refuses before writing unless at least max(`--mirror-keep-gib`, default 100 GiB, 10% of the
  volume) stays free afterwards. A mirror rebuilt in place counts its old size as free. Note: `--mirror-gib` larger
  than the model gives the whole expert set (153 GiB), and that is what gets written if it fits.
- `MAKE-DS41-MIRROR.bat [file] [GiB]` takes the model and `--rank` from the config, builds the mirror (default
  `C:\ai\ds41-mirror\V41-GSQ-1017.mirror`, 85 GiB, 41 s here) and then compares every mirrored expert with the model
  file (`--check-mirror`). At start the engine checks the mirror's size, header and 8 experts. If they do not match,
  it logs `blob mirror: NOT used (...)` and reads everything from the model file.

### Measured

`tools/v41/bench_long.py`'s files, a fresh server per run (the config plus `MIMO_CHUNK_PROF=1`), temperature 0, thinking
off. Long: cold 8K, 16K, warm 8K, 1,500 tokens out. Short: cold 2K, warm 2K, 4K, then two follow-up turns on the 4K
conversation (~310 new tokens each), 400 tokens out. Two runs of the long set each, alternating.

| long set | prefill 8K cold / 16K / 8K warm (tok/s) | prompt file-tier read (GB/s) | expert phase 8K / 16K / 8K (s) | decode (tok/s) | ms per verify pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| without the mirror, run 1 | 445 / 561 / 490 | 6.82-6.85 | 9.79 / 16.75 / 8.87 | 25.6 / 25.5 / 27.3 | 105 / 108 / 95 |
| without the mirror, run 2 | 457 / 600 / 480 | 6.82 | 9.48 / 15.99 / 9.14 | 28.0 / 26.3 / 25.9 | 97 / 101 / 94 |
| **mirror, share 0.5, run 1** | **498 / 641 / 515** | 10.8-11.3 | 8.12 / 14.17 / 7.93 | 28.0 / 26.0 / 27.1 | 96 / 100 / 95 |
| **mirror, share 0.5, run 2** | **504 / 647 / 522** | 11.1-11.5 | 7.98 / 14.03 / 7.83 | 28.2 / 26.5 / 26.8 | 96 / 100 / 94 |

| short set (prefill tok/s) | cold 2K | warm 2K | 4K | follow-up 1 / 2 (310 new tokens) |
| --- | ---: | ---: | ---: | ---: |
| without the mirror (2 runs) | 157 / 168 | 187 / 200 | 325 / 326 | 60.1 / 61.2 |
| **mirror, share 0.5** | **189** | **226** | **359** | 61.7 / 62.5 |

Share (one cold 8K request each, mirror on): 0.4 -> 10.3 GB/s, 490 tok/s; 0.5 -> 11.3-11.5 GB/s, 498-504; 0.6 ->
11.4 GB/s, 498. The config uses 0.5. The engine's default stays 0.28, measured on Linux with a mirror drive half as
fast as the model's.

The answers open with the same text as without the mirror (the bytes read are the same), and every mirrored expert
was compared with the model file after the build. The engine reads 10.4-11.5 GB/s, not the 13.1 GB/s the drives gave
`ds41_rdbench`. Not measured: why (the arena's DMA running alongside is one candidate), prompts above 16K, the
512K context filled, a mirror on a drive that shares the chipset link with the model, and whether the pagefile on
`C:` gets busy enough under memory pressure to slow the mirror (paging was not recorded).

## Chat turns re-read the whole prompt (fixed 2026-10-21)

Every follow-up turn of a thinking chat logged `reuse lcp=113191 ... snap=-1 prompt=115268 -> reused=0`. The
Monitor showed `REUSED 0`, and each turn spent about 150-175 s reading 113-115K tokens again. Cause: DeepSeek's encoder
ends a prompt with `<｜Assistant｜><think>`, and the next prompt renders that same turn as
`<｜Assistant｜></think>answer<｜end▁of▁sentence｜>` (past reasoning is dropped, as the model's template says). So the
new prompt matches all of the snapshot except its last token. The engine only resumed at `lcp >= snapshot`, so it
read everything again. The rendering is the model's own and stays as it is.

Fix: `Ds41Dense::restore_to(pos)` restores the prompt-end snapshot and continues a few tokens below it. This is exact
because every cache is indexed by position except the window ring and the ratio-2 compressor state. The window ring
holds `RAW = SWA + 16` positions, so it can go up to 16 tokens back. The ratio-2 ring holds the last 2 states, so an
odd `pos` can go back only 1. The log says `serve: rewound the N-token snapshot to lcp M`.

Measured (3 greedy turns over a 28.4K-token document, `--gpu2 1`, fast routing):

| turn | prompt | reused | prompt read |
|---|---:|---:|---:|
| 1 | 28,429 | 0 | 46.1 s |
| 2 | 28,573 | 28,428 | 4.4 s |
| 3 | 28,714 | 28,572 | 4.5 s |
| 1 (111.7K document, temperature 1) | 111,676 | 0 | 158.4 s |
| 2 | 111,806 | 111,675 | 4.5 s |

Exactness (`STRATA_SERVE_DUMP`, the last prompt row's logits for turn 2): rewound vs cold, mean |diff| 0.35 and KL
8e-4. Two cold reads of the same prompt differ by mean |diff| 0.47 and KL 3e-4. The engine is not bitwise
repeatable from run to run here, and the rewind falls inside that spread. All three runs pick the same top token
(p = 0.99).

### Chats with images

A prompt with an image did no reuse at all. The engine cleared its state and took no snapshot for every image
request, because each image is the same run of `image_token` ids, and comparing tokens alone would "continue" a
different picture. An agent chat with 4 screenshots in its first message therefore re-read 11.9K -> 19.9K tokens on
every step, with 0 reused. Now each image position is compared by a hash of its embedding row (the server caches
encoded images by the image's hash, so the same picture gives the same bytes). The same image matches, and a
different one ends the match where it starts. Image prompts are snapshotted like text ones.

Measured (a 14K-token prompt with 2 images in the first user message, 4 greedy turns): prompt read 28.3 s, then
2.5 / 3.0 / 0.9 s with 14,009 / 14,081 / 14,178 tokens reused. The same prompt with the two images swapped matched
only up to the first image (lcp 13,241) and was read again.

## Debugging options

- `-DSTRATA_DS41_DEBUGINFO=ON`: `/Z7` + a PDB, so the crash handler's stack has function names and lines.
- `-DSTRATA_DS41_ASAN=ON`: AddressSanitizer on the engine and the Strata libraries it links. Run it with the
  `clang_rt.asan_dynamic-x86_64.dll` folder from Visual Studio on `PATH`.
- `DS41_HEAP_CHECK=1`: validates every process heap at the load checkpoints and before each layer's expert tier.

## Image input (vision) on Windows (2026-10-21)

`strata-ds41-windows.json` now runs with `--vision` and a CPU image encoder (upstream #8, `docs/ds4/V41_VISION.md`):

- Encoder: `tools/v41/vision/ds41_vision.py` in its own venv (`H:/ai/xolot/venv-vision`, PyTorch 2.14.1+cpu, NumPy,
  Pillow), f32, 8 threads. It reads the vision/aligner weights from the DSpark sidecar `V41-MTP-Vision-1017.Q2_K.gguf`
  (the 1017 GGUF already carries every layer's `exp_probs_b_vl.bias`), and DeepSeek's `vision.py` /
  `image_processor.py` from the 1017 repo's `inference/` folder (`H:/ai/xolot/dsv41-inference/inference`).
- Windows fix: the sidecar reader used `os.pread`, which Python has no Windows version of; it now seeks and reads.
- `--dtype bf16` fails in the upstream script (modules stay f32), so it stays on f32.
- pip behind Avast's HTTPS scanning: `PIP_CERT="C:/ProgramData/Avast Software/Avast/wscert.pem"`.

Measured on this PC (RTX 5090 + 3090, 7800X3D):

| Check | Result |
| --- | --- |
| Encoder alone, DeepSeek's `carrots.jpeg` (444 image tokens) | 12.1-12.5 s per image, about 2 GB RAM |
| `check_api.py` (red square / blue circle / both / text after) | PASS, 0.9-9.7 s per request |
| `carrots.jpeg` with "describe in two sentences" | correct description, 24.0 s end to end |
| Text chat after that, 9,317-token prompt, then a follow-up | turn 2 reused 9,316 tokens, prompt read 23.4 s -> 2.0 s |

Limits (upstream's): an image request prefills its whole prompt and the next request starts fresh, so an image in a
long chat costs a full prompt read. The encoder uses the CPU only, and no VRAM.

## Keep-alive events for SDK clients (2026-10-22)

pi ("pi-ai stream idle timeout after 300000ms") cancelled six agent turns on 2026-10-21: each was a tool call writing
a large file (10k-21k tokens, 6-19 minutes). The server sends a tool call whole once it is complete, and until then
the stream carried only `: keep-alive` SSE comments, which SDK-based clients drop, so their idle timer ran out. The
OpenAI and Anthropic streams now send a real event at least every 15 s while nothing else goes out (`EVENT_BEAT_S`
in `serve/server.py`): a chunk with an empty delta (OpenAI), or Anthropic's own `ping` event. The Responses API
already did this for Codex.

| Test (thinking off, one long `write_file` call) | Longest gap between events |
| --- | --- |
| OpenAI, tool call held 71 s, 6,349 bytes of arguments, arrived intact | 20.1 s (4 empty-delta chunks) |
| OpenAI, held 254 s (hit max_tokens) | 20.2 s (13 empty-delta chunks) |
| Anthropic, held 92 s | 20.1 s (4 `ping` events) |

`serve/test_server.py` (314 tests) and the keep-alive tests pass.

## Forced tool_choice (2026-10-22)

A forced `tool_choice` (OpenAI `required` / a named function, Anthropic `any` / `tool`) used to open the call as
`<｜DSML｜ invoke
<function=NAME>` - the frontend's Qwen opening after the patched call tag - and the model's call came
back as text. `serve_ds41.py` now opens it the way DeepSeek's encoder writes one (`<｜DSML｜ calls>` then
`<｜DSML｜ invoke name="NAME">`). Checked live: OpenAI named (thinking off) and `required` (thinking on), Anthropic
`tool` and `any` all return a parsed tool call; `test_serve_ds41.py` passes. With a named tool the model may add a
second call after the forced one in the same block.
