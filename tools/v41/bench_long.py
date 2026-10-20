#!/usr/bin/env python3
"""bench_long.py - long-prompt / long-output runs against a running serve_ds41.py, cold and warm.

Each request's prompt is built from this repository's own source and docs (different files per request, so no two
requests share a prefix unless the case says so), cut to an exact token count with the model's tokenizer.  The numbers
come from the engine's own lines in its log (prefill tok/s, decode tok/s, expert sources) plus the client wall time.

  python tools/v41/bench_long.py --tokenizer <model>/tokenizer --log logs/strata-ds41.log [--port 8150]

Start the server fresh first: the first case is the cold one (empty VRAM expert cache).
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "v41"))
import strata_tokenizer as ST  # noqa: E402

ASK = ("\n\nAbove is source code and documentation from a project. Write a long, detailed technical explanation of "
       "what this code does: every component, how they fit together, and the important design decisions. Use headings "
       "and be thorough - at least 1500 words.")


def load_tok(tp: Path):
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tp / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tp / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types, pre="joyai-llm")


def corpus(files, tok, n_tokens):
    """The given files' text, cut to n_tokens tokens (the ASK suffix comes on top)."""
    parts, have = [], 0
    for f in files:
        t = f"\n\n===== {f.relative_to(ROOT).as_posix()} =====\n" + f.read_text(encoding="utf-8", errors="replace")
        ids = tok.encode(t)
        if have + len(ids) >= n_tokens:
            parts.append(tok.decode(ids[: n_tokens - have]))
            break
        parts.append(t)
        have += len(ids)
    return "".join(parts)


def post(port, messages, max_tokens):
    body = json.dumps({"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "reasoning_effort": "none"}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def log_lines(log: Path, start: int):
    with open(log, "rb") as f:
        f.seek(start)
        return f.read().decode("utf-8", errors="replace").replace("\r", "\n").split("\n")


def parse(lines):
    out = {}
    for ln in lines:
        m = re.match(r"serve: prompt (\d+) tokens \((\d+) reused.*?\) in ([\d.]+) s = ([\d.]+) tok/s "
                     r"\(experts (\d+) hit / (\d+) cpu / (\d+) pcie / (\d+) file\)", ln)
        if m:
            out.update(p_tok=int(m[1]), p_reused=int(m[2]), p_s=float(m[3]), p_tps=float(m[4]),
                       p_ex=tuple(int(m[i]) for i in range(5, 9)))
        m = re.match(r"serve: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \((\w+)\); experts: (\d+) hit, (\d+) cpu, "
                     r"(\d+) pcie, (\d+) file", ln)
        if m:
            out.update(d_tok=int(m[1]), d_s=float(m[2]), d_tps=float(m[3]), stop=m[4],
                       d_ex=tuple(int(m[i]) for i in range(5, 9)))
        m = re.match(r"serve: DSpark (\d+) passes, ([\d.]+) tokens/pass", ln)
        if m:
            out.update(tpp=float(m[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--log", default=str(ROOT / "logs" / "strata-ds41.log"))
    ap.add_argument("--port", type=int, default=8150)
    ap.add_argument("--out", default=str(ROOT / "logs" / "bench_long.json"))
    a = ap.parse_args()
    tok = load_tok(Path(a.tokenizer))
    log = Path(a.log)
    src = sorted((ROOT / "src").rglob("*.cpp")) + sorted((ROOT / "src").rglob("*.cu"))
    docs = sorted((ROOT / "docs").glob("*.md"))
    v41 = sorted((ROOT / "tools" / "v41" / "engine").glob("*.cpp"))
    serve = sorted((ROOT / "serve").glob("*.py"))
    # (name, files, prompt tokens, max output tokens): every case reads different files, so none shares a prefix
    cases = [
        ("cold 2K", docs[:40], 2000, 1500),
        ("8K", v41, 8000, 1500),
        ("16K", src[: len(src) // 2], 16000, 1500),
        ("32K", serve, 32000, 1500),
        ("warm 2K (new text)", docs[40:], 2000, 1500),
        ("warm 8K (new text)", src[len(src) // 2:], 8000, 1500),
    ]
    results, convo32 = [], None
    for name, files, n, mx in cases:
        text = corpus(files, tok, n) + ASK
        msgs = [{"role": "user", "content": text}]
        start = log.stat().st_size
        d, wall = post(a.port, msgs, mx)
        time.sleep(1.0)
        r = {"case": name, "wall_s": round(wall, 1), **parse(log_lines(log, start))}
        r["usage"] = d.get("usage")
        results.append(r)
        print(json.dumps(r), flush=True)
        if name == "32K":
            convo32 = msgs + [{"role": "assistant", "content": d["choices"][0]["message"]["content"]}]
    # an agent-style follow-up turn on the 32K conversation: the prefix is reused, only the new turn is prefilled
    msgs = convo32 + [{"role": "user", "content": "Now list the five most important functions you described, with "
                                                  "one paragraph each on how they could be made faster."}]
    start = log.stat().st_size
    d, wall = post(a.port, msgs, 1500)
    time.sleep(1.0)
    r = {"case": "follow-up on 32K", "wall_s": round(wall, 1), **parse(log_lines(log, start)), "usage": d.get("usage")}
    results.append(r)
    print(json.dumps(r), flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
