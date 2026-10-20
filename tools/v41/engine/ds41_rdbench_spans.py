#!/usr/bin/env python3
"""ds41_rdbench_spans.py - the expert spans ds41_rdbench reads, from the model's <gguf>.experts.json.

Writes a binary list of blobs, each 3 x {u64 offset, u64 bytes} (gate, up, down), in the order the engine's chunk
prefetch reads them: layer by layer, ascending expert id, a `--share` fraction of each layer's experts (the file tier;
the rest would sit in VRAM or the RAM arena).

  python tools/v41/engine/ds41_rdbench_spans.py <model>.gguf.experts.json spans.bin [--share 0.5] [--seed 1]
"""
import argparse
import json
import random
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experts_json")
    ap.add_argument("out")
    ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    d = json.load(open(a.experts_json))
    rnd = random.Random(a.seed)
    out, total = [], 0
    for lk in sorted(d["layers"], key=int):
        L = d["layers"][lk]
        off = {int(e): v for e, v in L["off"].items()}
        # a span's bytes = the stride between neighbouring experts of the same type in the same role tensor
        size = {}
        for role, tkey in (("gate", "gu_type"), ("up", "gu_type"), ("down", "d_type")):
            by_t = {}
            for e, v in off.items():
                by_t.setdefault(L[tkey][e], []).append(v[role])
            for t, vs in by_t.items():
                vs.sort()
                st = min(b - a for a, b in zip(vs, vs[1:])) if len(vs) > 1 else None
                size[(role, t)] = st
        for e in sorted(off):
            if rnd.random() >= a.share:
                continue
            spans = []
            for role, tkey in (("gate", "gu_type"), ("up", "gu_type"), ("down", "d_type")):
                n = size[(role, L[tkey][e])]
                if n is None:
                    break
                spans.append((off[e][role], n))
            if len(spans) == 3:
                out.append(spans)
                total += sum(n for _, n in spans)
    with open(a.out, "wb") as f:
        for s in out:
            for o, n in s:
                f.write(struct.pack("<QQ", o, n))
    print(f"{len(out)} blobs, {total / 2**30:.1f} GiB, mean blob {total / max(1, len(out)) / 2**20:.2f} MiB -> {a.out}")


if __name__ == "__main__":
    main()
