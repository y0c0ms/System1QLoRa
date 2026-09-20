#!/usr/bin/env python3
"""Exact sequence scoring: logP(option_text | prompt), by teacher forcing.

WHY LETTER SCORING HAD TO GO. It failed three ways, all measured:
  * options past Z have no letter, so 230 of 1751 test rows were simply dropped
  * a letter can fall outside `top_n` and take a sentinel - 26.5% of ag_news rows
    for 3.6, which made it predict "World" on 82.5% against a uniform truth
  * after "Answer:" a raw continuation wants prose, so on some tasks the letters
    sink out of the window entirely and on others they do not - which is why the
    SAME model scored 0.8900 on ag_news via the chat endpoint and 0.3750 via
    /completion

TWO MECHANISMS THAT DO NOT WORK, so nobody retries them:
  1. GBNF grammar forcing the option string. llama.cpp matches the grammar
     CHARACTER-wise, so " World" came back as ['Wo','rl','d'] and "Sci/Tech" as
     ['S','ci','/T','ech']. Those logprobs describe an unnatural token path
     (~-11/token) and are not logP(option).
  2. `/v1/completions` with `echo: true, logprobs: 1`. Accepted, returns zero
     tokens. llama.cpp does not implement prompt-token logprobs.

WHAT DOES WORK. `/completion` accepts a TOKEN-ID ARRAY as the prompt, and each
entry in `top_logprobs` carries its token `id`. So for option tokens t0..tn we
send prefix+t0..t(i-1) and look up t(i) BY ID in the returned distribution. Exact,
natural tokenisation, and when a token falls outside the window we know it rather
than guessing.

COST is one forward pass per option token; `cache_prompt` makes the shared prefix
free after the first call, so parallelism is over ROWS (keeping a slot's cache
warm), never over the calls within a row.

BOTH total and length-normalised (mean) logprob are reported. Total favours short
options; mean favours long ones. Neither is obviously right, so both ship.
"""

import argparse
import collections
import json
import math
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
# A token outside the top-N has logprob AT MOST that of the Nth candidate, so the
# window's own minimum is a principled upper bound. An arbitrary constant (-20.0
# in the first draft) is not: it can sit far above or far below the true value
# depending on how peaked the distribution is, and 14.68% of lookups hit it.
FLOOR_LP = -25.0         # only used if the window comes back empty
MAXOPT = [0]


class Client:
    """One llama.cpp client, with a SLOT PINNED PER WORKER THREAD.

    Without pinning, four workers scoring four different rows land on whichever
    llama.cpp slot is free, so each request evicts another worker's KV cache and
    every call re-prefills the whole prompt. Measured: 16 rows in ten minutes,
    ~650 ms/call, which extrapolated to ~15 HOURS for the val split alone.

    With `id_slot` fixed per thread, a worker's growing prompt always meets its
    own cache: 2,071 ms on the cold call, then ~100 ms.
    """

    def __init__(self, base, top_n, n_slots=4):
        self.base, self.top_n = base, top_n
        self.n_slots = n_slots
        self._tok = {}
        self._lock = threading.Lock()
        self._slots = {}
        self._next = [0]
        self._local = threading.local()

    def slot(self):
        s = getattr(self._local, "slot", None)
        if s is None:
            with self._lock:
                s = self._next[0] % self.n_slots
                self._next[0] += 1
            self._local.slot = s
        return s

    def _post(self, path, body, timeout=300):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def tokenize(self, text):
        """Cached: banking77 reuses the same 77 option strings across 150 rows."""
        with self._lock:
            hit = self._tok.get(text)
        if hit is not None:
            return hit
        ids = self._post("/tokenize", {"content": text}).get("tokens", [])
        with self._lock:
            self._tok[text] = ids
        return ids

    def logprob_of(self, prompt_ids, target_id):
        """(logprob, hit) for target_id. On a miss, returns the window minimum.

        The window minimum is an UPPER BOUND on a missing token's logprob, so a
        wrong option can only be flattered, never unfairly punished.
        """
        body = {"prompt": prompt_ids, "n_predict": 1, "n_probs": self.top_n,
                "temperature": 1.0, "top_k": 0, "top_p": 1.0, "cache_prompt": True,
                "id_slot": self.slot()}
        d = self._post("/completion", body)
        cp = d.get("completion_probabilities") or []
        if not cp:
            return FLOOR_LP, False
        top = cp[0].get("top_logprobs") or []
        lo = FLOOR_LP
        for e in top:
            if e.get("id") == target_id:
                return e.get("logprob"), True
            lp = e.get("logprob")
            if lp is not None and lp < lo or lo == FLOOR_LP:
                lo = min(lo, lp) if lp is not None else lo
        return lo, False


def build_prefix(row):
    # No option list: the options are what we are scoring, not context.
    return "State:\n%s\n\nQuestion:\n%s\n\nAnswer:" % (row["state"], row["question"])


def subset_options(row, max_options, seed=0):
    """Gold plus sampled distractors, when the full set is too expensive.

    WHY THIS EXISTS AND WHAT IT COSTS. Sequence scoring is one forward pass per
    option TOKEN. banking77 is 77 options ~ 258 lookups per row; at the measured
    ~100 ms/lookup the full test split is 73,594 lookups = 2 hours PER MODEL, and
    concurrency does not help because --n-cpu-moe 20 puts a fifth of 3.6's experts
    on CPU, so parallel forward passes contend.

    Capping makes banking77 and tickets_queue EASIER than they were under letter
    scoring, where the full set was used. Any number from a capped run must say so.
    Upstream caps at 8 for training for the same reason; this is eval, so it is a
    deliberate trade, not a default.

    Returns (options, answer_index) with the gold guaranteed present and the
    position shuffled under a fixed seed, so position is not a signal.
    """
    opts, ans = list(row["options"]), int(row["answer_index"])
    if not max_options or len(opts) <= max_options:
        return opts, ans
    import random
    rng = random.Random(seed * 100003 + hash(row["state"][:64]) % 100003)
    others = [i for i in range(len(opts)) if i != ans]
    rng.shuffle(others)
    keep = [ans] + others[: max_options - 1]
    rng.shuffle(keep)
    return [opts[i] for i in keep], keep.index(ans)


def score_row(cli, row, max_options=0):
    row = dict(row)
    row["options"], row["answer_index"] = subset_options(row, max_options)
    prefix_ids = cli.tokenize(build_prefix(row))
    totals, means, missing, lookups = [], [], 0, 0
    for opt in row["options"]:
        opt_ids = cli.tokenize(" " + str(opt))
        if not opt_ids:
            totals.append(FLOOR_LP); means.append(FLOOR_LP); missing += 1
            lookups += 1
            continue
        lp_sum, n_miss = 0.0, 0
        ctx = list(prefix_ids)
        for tid in opt_ids:
            lookups += 1
            lp, hit = cli.logprob_of(ctx, tid)
            if not hit:
                n_miss += 1
            lp_sum += lp
            ctx.append(tid)
        missing += n_miss
        totals.append(lp_sum)
        means.append(lp_sum / len(opt_ids))
    # Per-TOKEN accounting. A banking77 row makes ~231 lookups, so "this row had
    # at least one miss" is near-certain and tells you nothing; the rate does.
    return totals, means, missing, lookups, row["answer_index"]


def softmax(v, t=1.0):
    m = max(x / t for x in v)
    e = [math.exp(x / t - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def ece(probs, answers, n_bins=10):
    conf = [max(p) for p in probs]
    corr = [p.index(max(p)) == a for p, a in zip(probs, answers)]
    n, out = len(probs), 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j in range(n) if lo < conf[j] <= hi]
        if not idx:
            continue
        out += (len(idx) / n) * abs(sum(corr[j] for j in idx) / len(idx)
                                    - sum(conf[j] for j in idx) / len(idx))
    return out


def brier(probs, answers):
    return sum(sum((x - (1.0 if i == a else 0.0)) ** 2 for i, x in enumerate(p))
               for p, a in zip(probs, answers)) / len(probs)


def fit_T(vecs, answers):
    best, bnll = 1.0, float("inf")
    for i in range(200):
        t = 0.05 * (1.0353 ** i)
        nll = 0.0
        for v, a in zip(vecs, answers):
            nll -= math.log(max(softmax(v, t)[a], 1e-12))
        if nll < bnll:
            bnll, best = nll, t
    return best, (best <= 0.0501 or best >= 0.05 * (1.0353 ** 199) * 0.999)


def run(cli, rows, ckpt, workers, tag):
    have = {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            have[d["i"]] = d
        print("  resuming: %d rows already scored" % len(have), flush=True)
    todo = [(i, r) for i, r in enumerate(rows) if i not in have]
    print("  %s: %d rows, %d to do" % (tag, len(rows), len(todo)), flush=True)
    lock, done, t0 = threading.Lock(), [0], time.time()
    fh = open(ckpt, "a")

    def work(pair):
        i, r = pair
        try:
            tot, mean, miss, look, ans = score_row(cli, r, MAXOPT[0])
            rec = {"i": i, "total": tot, "mean": mean, "missing": miss,
                   "lookups": look, "ans": ans}
        except Exception as e:  # noqa: BLE001 - a failed row is recorded, never fatal
            rec = {"i": i, "total": None, "mean": None, "missing": len(r["options"]),
                   "lookups": len(r["options"]), "err": type(e).__name__}
        with lock:
            done[0] += 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if done[0] % 25 == 0:
                el = time.time() - t0
                print("    %d/%d  (%.1f rows/min, eta %.0f min)"
                      % (done[0], len(todo), done[0] / el * 60,
                         (len(todo) - done[0]) / max(done[0] / el, 1e-9) / 60), flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    merged = dict(have)
    for line in open(ckpt):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        merged[d["i"]] = d
    return [dict(merged[i], row=r) for i, r in enumerate(rows)
            if i in merged and merged[i].get("total")]


def report(recs, key, T, tag):
    vecs = [r[key] for r in recs]
    ans = [r.get("ans", r["row"]["answer_index"]) for r in recs]
    probs = [softmax(v, T) for v in vecs]
    acc = sum(p.index(max(p)) == a for p, a in zip(probs, ans)) / len(recs)
    print("\n  === %s [%s] T=%.3f ===" % (tag, key, T))
    print("    n=%d  accuracy=%.4f  ECE=%.4f  Brier=%.4f  mean_conf=%.4f"
          % (len(recs), acc, ece(probs, ans), brier(probs, ans),
             sum(max(p) for p in probs) / len(recs)))
    by = collections.defaultdict(list)
    for r, p in zip(recs, probs):
        by[r["row"]["task"]].append((p, r.get("ans", r["row"]["answer_index"])))
    for t in sorted(by):
        ps = [x[0] for x in by[t]]; aa = [x[1] for x in by[t]]
        print("      %-18s n=%-5d acc=%.4f ece=%.4f"
              % (t, len(ps), sum(p.index(max(p)) == a for p, a in zip(ps, aa)) / len(ps),
                 ece(ps, aa)))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="rows per task (0 = all)")
    ap.add_argument("--max-options", type=int, default=0,
                    help="cap the option set (gold + sampled distractors). 0 = full sets. "
                         "Capping makes 77-option tasks EASIER - report it.")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    MAXOPT[0] = a.max_options
    cli = Client(a.base, a.top_n, n_slots=a.workers)

    def cap(rows):
        if not a.limit:
            return rows
        seen, out = collections.Counter(), []
        for r in rows:
            if seen[r["task"]] >= a.limit:
                continue
            seen[r["task"]] += 1
            out.append(r)
        return out

    val = cap(json.load(open(os.path.join(DATA, "val.json"))))
    test = cap(json.load(open(os.path.join(DATA, "test.json"))))
    print("base=%s tag=%s top_n=%d workers=%d" % (a.base, a.tag, a.top_n, a.workers))

    print("\nVAL (temperature fitted here)")
    vr = run(cli, val, os.path.join(RES, "seq_%s_val.jsonl" % a.tag), a.workers, "val")
    print("\nTEST")
    tr = run(cli, test, os.path.join(RES, "seq_%s_test.jsonl" % a.tag), a.workers, "test")

    out = {"tag": a.tag, "base": a.base, "top_n": a.top_n}
    for key in ("total", "mean"):
        T, sat = fit_T([r[key] for r in vr],
                       [r.get("ans", r["row"]["answer_index"]) for r in vr])
        if sat:
            print("\n  WARNING: temperature search saturated for '%s'" % key)
        report(vr, key, T, "VAL")
        out[key] = {"T": T, "test_acc": report(tr, key, T, "TEST")}
    miss_rows = sum(1 for r in tr if r["missing"])
    miss_tok = sum(r.get("missing", 0) for r in tr)
    look_tok = sum(r.get("lookups", 0) for r in tr)
    print("\n  token lookups: %d | outside the %d-window: %d (%.2f%%)"
          % (look_tok, a.top_n, miss_tok, 100.0 * miss_tok / max(look_tok, 1)))
    print("  rows touched by at least one miss: %d / %d (%.1f%%) - near-certain for "
          "77-option rows, so read the TOKEN rate above"
          % (miss_rows, len(tr), 100.0 * miss_rows / len(tr)))
    out["missing_tokens"] = miss_tok
    out["lookups"] = look_tok
    json.dump(out, open(os.path.join(RES, "seq_%s.json" % a.tag), "w"), indent=1)


if __name__ == "__main__":
    main()
