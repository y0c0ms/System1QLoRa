#!/usr/bin/env python3
"""T2: score options by letter-logprob against a model we already serve.

THE QUESTION THIS ANSWERS. system-one-gemma fine-tunes a 270M model with a
scoring head. Is that worth anything over prompting a model already resident on
the GPU and reading the probability it assigns to each option letter? If the
zero-training baseline matches it, the fine-tune is moot.

The method is upstream's own `run_baseline` (render options as A./B./C., read the
next-token logits of the letters, softmax those) ported to HTTP so it needs no
torch. That makes it a fair comparator rather than a different experiment.

METRICS mirror system_one.py exactly - 10 equal-width bins, top-1 confidence ECE;
Brier summed over options and NOT normalised for option count; temperature by
grid search minimising NLL over [0.25, 6.0] in 116 steps. Same definitions, so
the numbers are directly comparable to upstream's.

TEMPERATURE IS FITTED ON VAL AND APPLIED TO TEST. This is the thing the released
scorer got wrong: its shipped metrics.json fitted T on the very rows it scored,
which makes ECE optimistic by construction.

MISSING OPTIONS ARE NOT SILENTLY DROPPED. top_logprobs truncates; see AGENTS.md
rule 2. Every row records how many of its options failed to appear, and the
summary reports it. A renormalisation over a truncated option set is a wrong
number that looks right.
"""

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("LLAMA_SERVER_URL", "http://localhost:11435")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
ENDPOINT = ["chat"]      # set from argv in main()
BASE_URL = [BASE]


def build_prompt(row):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(row["options"]))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (row["state"], row["question"], opts))


def score_row_completion(row, model, top_n, base, timeout=300):
    """Score via llama.cpp's RAW /completion endpoint - no chat template.

    WHY THIS EXISTS. Qwen3.8-27B cannot disable reasoning, and the reasoning is
    injected by the CHAT TEMPLATE. Through /v1/chat/completions the first
    generated token on a real scoring prompt was 'We' and no option letter
    appeared in the top 20 - letter scoring reads position 0, so it cannot work.
    /completion applies no template, so nothing is injected and the letters are
    right there.

    BOTH models must be run through THIS path, not just 3.8. Comparing a
    chat-endpoint 3.6 against a completion-endpoint 3.8 would confound the model
    with the prompt format.

    The response shape is `completion_probabilities[0].top_logprobs`, entries
    {token, logprob, bytes, id}. An earlier probe guessed `top_probs`/`probs`,
    got an empty list, and still reported the route usable.
    """
    payload = {"prompt": build_prompt(row), "n_predict": 1, "n_probs": top_n,
               "temperature": 1.0, "top_k": 0, "top_p": 1.0, "cache_prompt": False}
    if model:
        payload["model"] = model
    req = urllib.request.Request(base + "/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
    except Exception:  # noqa: BLE001
        return None, len(row["options"]), (time.time() - t0) * 1000
    ms = (time.time() - t0) * 1000
    cp = body.get("completion_probabilities") or []
    if not cp:
        return None, len(row["options"]), ms
    top = cp[0].get("top_logprobs") or []
    best = {}
    for e in top:
        t = str(e.get("token", "")).strip()      # letters arrive as ' A', not 'A'
        if len(t) == 1 and t in LETTERS:
            lp = e.get("logprob")
            if lp is not None and (t not in best or lp > best[t]):
                best[t] = lp
    k = len(row["options"])
    raw = [best.get(LETTERS[i]) for i in range(k)]
    missing = sum(1 for x in raw if x is None)
    if missing == k:
        return None, missing, ms
    return [(-60.0 if x is None else x) for x in raw], missing, ms


def score_row(row, model, top_n, timeout=120):
    """Return (probs over options, missing count, latency_ms) or (None, k, ms)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(row)}],
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": top_n,
        # Qwen3.6 is a thinking model. Left on, the first token is reasoning
        # preamble and no option letter is ever in position 0.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
    except Exception:  # noqa: BLE001 - a failed row is recorded, never fatal
        return None, len(row["options"]), (time.time() - t0) * 1000
    ms = (time.time() - t0) * 1000
    try:
        top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None, len(row["options"]), ms

    # Keep the HIGHEST logprob per letter: a letter can surface as both 'A' and
    # ' A' depending on tokenisation, and taking whichever came first would
    # depend on server ordering.
    best = {}
    for e in top:
        t = e["token"].strip()
        if len(t) == 1 and t in LETTERS:
            lp = e["logprob"]
            if t not in best or lp > best[t]:
                best[t] = lp

    k = len(row["options"])
    raw = [best.get(LETTERS[i]) for i in range(k)]
    missing = sum(1 for x in raw if x is None)
    if missing == k:
        return None, missing, ms
    # An option the server never reported gets -inf, not a guess. It still
    # occupies its slot so the distribution has the right arity.
    logits = [(-60.0 if x is None else x) for x in raw]
    return logits, missing, ms


def softmax(logits, t=1.0):
    m = max(x / t for x in logits)
    e = [math.exp(x / t - m) for x in logits]
    s = sum(e)
    return [x / s for x in e]


def ece(probs, answers, n_bins=10):
    conf = [max(p) for p in probs]
    correct = [p.index(max(p)) == a for p, a in zip(probs, answers)]
    n, out = len(probs), 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [j for j in range(n) if (conf[j] > lo and conf[j] <= hi)]
        if not idx:
            continue
        acc = sum(correct[j] for j in idx) / len(idx)
        av = sum(conf[j] for j in idx) / len(idx)
        out += (len(idx) / n) * abs(acc - av)
    return out


def brier(probs, answers):
    tot = 0.0
    for p, a in zip(probs, answers):
        tot += sum((x - (1.0 if i == a else 0.0)) ** 2 for i, x in enumerate(p))
    return tot / len(probs)


def fit_temperature(logit_sets, answers):
    best_t, best_nll = 1.0, float("inf")
    for i in range(116):
        t = 0.25 + i * 0.05
        nll = 0.0
        for L, a in zip(logit_sets, answers):
            p = softmax(L, t)
            nll -= math.log(max(p[a], 1e-12))
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def run_split(rows, model, top_n, workers, ckpt=None):
    """Score a split, appending every row to `ckpt` as it lands.

    WHY CHECKPOINTING IS NOT OPTIONAL HERE. The first full attempt ran 35 minutes
    against a contended GPU, stalled, and was killed - and because results lived
    only in memory, all 35 minutes were lost. A run against a shared GPU WILL be
    interrupted; the only question is whether it costs everything.

    Rows are keyed by their index in the split, so a resumed run skips exactly
    what is already on disk and nothing is scored twice.
    """
    scoreable = [(i, r) for i, r in enumerate(rows) if len(r["options"]) <= 26]
    skipped = len(rows) - len(scoreable)

    have = {}
    if ckpt and os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue  # a torn final line from a kill mid-write
                have[d["i"]] = d
        print("  resuming: %d rows already scored in %s" % (len(have), os.path.basename(ckpt)))

    todo = [(i, r) for i, r in scoreable if i not in have]
    print("  %d rows, %d scoreable by letter, %d skipped (>26 options), %d to do"
          % (len(rows), len(scoreable), skipped, len(todo)))

    lock = __import__("threading").Lock()
    done = [0]
    fh = open(ckpt, "a") if ckpt else None

    def work(pair):
        i, r = pair
        if ENDPOINT[0] == "completion":
            logits, missing, ms = score_row_completion(r, model, top_n, BASE_URL[0])
        else:
            logits, missing, ms = score_row(r, model, top_n)
        rec = {"i": i, "logits": logits, "missing": missing, "ms": ms}
        with lock:
            done[0] += 1
            if fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()  # survive a kill -9, not just a clean exit
            if done[0] % 25 == 0:
                print("    %d/%d  (%.1f rows/min)"
                      % (done[0], len(todo), done[0] / max(1e-9, time.time() - t0) * 60),
                      flush=True)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    wall = time.time() - t0
    if fh:
        fh.close()
    print("    %d scored this pass in %.1fs" % (len(todo), wall), flush=True)

    # Re-read the checkpoint so this pass and every previous pass are merged.
    merged = dict(have)
    if ckpt and os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                merged[d["i"]] = d
    out = [{"row": r, "logits": merged[i]["logits"], "missing": merged[i]["missing"],
            "ms": merged[i]["ms"]} for i, r in scoreable if i in merged]
    return out, skipped, wall


def take_per_task(rows, n):
    """Cap rows PER TASK, never globally.

    The corpus is ordered by task and banking77 comes first, so `rows[:40]` is
    40 banking77 rows - all of which have 77 options and are unscoreable by
    letter. A global cap does not sample the benchmark, it selects one task.
    """
    seen, out = {}, []
    for r in rows:
        t = r["task"]
        if seen.get(t, 0) >= n:
            continue
        seen[t] = seen.get(t, 0) + 1
        out.append(r)
    return out


def summarise(records, tag, temperature=1.0):
    ok = [r for r in records if r["logits"] is not None]
    failed = len(records) - len(ok)
    if not ok:
        print("\n  === %s ===\n    NOTHING SCORED (%d rows, all failed or unscoreable)"
              % (tag, len(records)))
        return None
    any_missing = sum(1 for r in ok if r["missing"] > 0)
    probs = [softmax(r["logits"], temperature) for r in ok]
    answers = [r["row"]["answer_index"] for r in ok]
    acc = sum(p.index(max(p)) == a for p, a in zip(probs, answers)) / len(ok)
    print("\n  === %s (T=%.2f) ===" % (tag, temperature))
    print("    scored            : %d   (failed %d)" % (len(ok), failed))
    print("    rows with a MISSING option: %d  (%.1f%%)"
          % (any_missing, 100.0 * any_missing / len(ok)))
    print("    accuracy          : %.4f" % acc)
    print("    ECE (10 bins)     : %.4f" % ece(probs, answers))
    print("    Brier             : %.4f" % brier(probs, answers))
    print("    mean confidence   : %.4f" % (sum(max(p) for p in probs) / len(ok)))
    by = {}
    for r, p in zip(ok, probs):
        by.setdefault(r["row"]["task"], []).append((p, r["row"]["answer_index"]))
    print("    per task:")
    for t in sorted(by):
        ps = [x[0] for x in by[t]]
        ans = [x[1] for x in by[t]]
        a = sum(p.index(max(p)) == y for p, y in zip(ps, ans)) / len(ps)
        print("      %-18s n=%-5d acc=%.4f ece=%.4f" % (t, len(ps), a, ece(ps, ans)))
    return {"n": len(ok), "failed": failed, "missing_rows": any_missing,
            "accuracy": acc, "ece": ece(probs, answers), "brier": brier(probs, answers),
            "temperature": temperature}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6-35b")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--endpoint", choices=["chat", "completion"], default="chat",
                    help="completion = raw llama.cpp, no chat template. Required for "
                         "any model whose reasoning cannot be disabled.")
    ap.add_argument("--base", default=BASE, help="server root, e.g. http://localhost:11435")
    ap.add_argument("--tag", default=None, help="name for the results/checkpoint files")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows PER TASK (0 = all). Never a global slice - see take_per_task.")
    ap.add_argument("--data-dir", default=DATA, help="dir holding val.json/test.json")
    a = ap.parse_args()

    val = json.load(open(os.path.join(a.data_dir, "val.json")))
    test = json.load(open(os.path.join(a.data_dir, "test.json")))
    if a.limit:
        val, test = take_per_task(val, a.limit), take_per_task(test, a.limit)

    ENDPOINT[0] = a.endpoint
    BASE_URL[0] = a.base
    tag = a.tag or a.model
    print("model=%s endpoint=%s base=%s top_logprobs=%d workers=%d"
          % (a.model, a.endpoint, a.base, a.top_n, a.workers))
    res_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(res_dir, exist_ok=True)
    ck = lambda s: os.path.join(res_dir, "ckpt_%s_%s.jsonl" % (tag, s))  # noqa: E731

    print("\nVAL (temperature is fitted here)")
    vrec, vskip, vwall = run_split(val, a.model, a.top_n, a.workers, ck("val"))
    vok = [r for r in vrec if r["logits"] is not None]
    T = fit_temperature([r["logits"] for r in vok],
                        [r["row"]["answer_index"] for r in vok])
    print("  fitted temperature: %.2f" % T)
    summarise(vrec, "VAL uncalibrated", 1.0)
    summarise(vrec, "VAL calibrated", T)

    print("\nTEST (temperature applied, never fitted here)")
    trec, tskip, twall = run_split(test, a.model, a.top_n, a.workers, ck("test"))
    summarise(trec, "TEST uncalibrated", 1.0)
    res = summarise(trec, "TEST calibrated", T)

    lat = sorted(r["ms"] for r in trec if r["logits"] is not None)
    print("\n  === latency (UNDER %d-WAY CONCURRENCY - not a single-stream figure) ===" % a.workers)
    print("    n=%d  p50=%.0f ms  p90=%.0f ms  p99=%.0f ms  max=%.0f ms"
          % (len(lat), lat[len(lat) // 2], lat[int(len(lat) * .9)],
             lat[int(len(lat) * .99)], lat[-1]))
    print("    wall for %d rows: %.1fs -> %.1f decisions/sec" % (len(lat), twall, len(lat) / twall))

    os.makedirs(os.path.join(DATA, "..", "results"), exist_ok=True)
    with open(os.path.join(DATA, "..", "results", "baseline_%s.json" % tag), "w") as f:
        json.dump({"model": a.model, "top_n": a.top_n, "workers": a.workers,
                   "skipped_gt26_options": {"val": vskip, "test": tskip},
                   "test": res, "latency_ms": {"p50": lat[len(lat) // 2],
                                               "p90": lat[int(len(lat) * .9)],
                                               "p99": lat[int(len(lat) * .99)]}}, f, indent=1)


if __name__ == "__main__":
    main()
