# ISTA IQ2_XXS vs our GSQ model (2026-10-24)

Same PC, same script (`tools/v41/bench_long.py`), fresh server each, 1,500 tokens out, thinking off.

## Speed (tokens per second, higher is better)

| Case | Prefill ours | Prefill ISTA | Decode ours | Decode ISTA |
|---|---:|---:|---:|---:|
| 2K prompt, cold | 225 | 240 | 30.4 | 42.4 |
| 8K prompt | **531** | 427 | 32.5 | **42.1** |
| 16K prompt | **674** | 553 | 32.3 | **39.2** |
| 32K prompt | **812** | 632 | 31.5 | **43.5** |
| 2K prompt, warm | 238 | **278** | 33.2 | **47.0** |
| 8K prompt, warm | 542 | 496 | 31.2 | **45.2** |
| Follow-up on 32K chat | 195 | **254** | 27.8 | **37.7** |

That table is ISTA without the second-SSD mirror (ours had it). 2026-10-25, both with the mirror on C: (ISTA's 64 GiB,
`MAKE-DS41-MIRROR.bat C:\ai\ds41-mirror\ISTA-IQ2_XXS.mirror 64 tools\v41\serve\strata-ds41-ista-windows.json`), two
fresh-server runs each:

| Case | Prefill ours | Prefill ISTA + mirror | Decode ours | Decode ISTA + mirror |
|---|---:|---:|---:|---:|
| 2K prompt, cold | 210-220 | **277-289** | 28-30 | **46-48** |
| 8K prompt | 521-522 | **526-532** | 32-34 | **45-46** |
| 16K prompt | 647-656 | 657-664 | 29-31 | **43-44** |
| 32K prompt | **782-798** | 746-757 | 25-32 | **43-45** |
| 2K prompt, warm | 220-233 | **306-311** | 31-33 | **45-48** |
| 8K prompt, warm | 525-535 | 523-548 | 28-31 | **41-44** |
| Follow-up on 32K chat | 172-191 | **247-252** | 27-29 | **36-38** |

With the mirror ISTA reads prompts as fast as ours (32K: 5% slower) and writes 40-60% faster.

## Quality (perplexity, 4096 tokens, lower is better)

| Text | Ours | ISTA |
|---|---:|---:|
| code | **2.83** | 2.89 |
| prose | **3.68** | 3.74 |
| C++ | **5.06** | 5.18 |

## Other

| | Ours | ISTA |
|---|---|---|
| Size | 174 GB | 138 GB |
| Load time | ~92 s | ~54 s |
| Image input | yes | no |
| Tool-call tests | pass | pass |

Start ours: `START-DS41.bat`. Start ISTA: `START-DS41-ISTA.bat`. Run one at a time (both use port 8150).
