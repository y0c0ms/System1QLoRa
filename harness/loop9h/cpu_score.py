#!/usr/bin/env python3
"""Small-hardware check: score decisions with llama.cpp on CPU from a GGUF and
compare them row-by-row with the GPU evaluation of the same model.

Readout: /completion with n_predict=1 and n_probs=N returns the top-N token
log-probs at the answer position (pre-sampling). The option letters' log-probs
are taken from that list by token id. AGENTS.md rule 2: if ANY option letter is
absent from the top-N the row is recorded as `missing` - never renormalized over
the letters that happened to be returned.

Latency (AGENTS.md rule 4 - figures ship with conditions): client-side wall time
per decision (prefill + one token, cache_prompt=false), warm (first 5 discarded),
fixed thread count, one request at a time; CPU model, threads, quantization,
prompt-token stats and /proc/loadavg are recorded next to the numbers.
"""

import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer
from eval_decider import text_field  # canonical field rendering; see eval_decider.text_field

ROOT = Path(__file__).resolve().parents[2]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_prompt(state, question, options):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(options))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (text_field(state), text_field(question), opts))


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * (len(xs) - 1) + 0.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--server", required=True, help="path to llama-server binary")
    ap.add_argument("--tokenizer", required=True, help="HF dir/id with the same vocab")
    ap.add_argument("--gpu-tag", default="", help="results/evals/<tag> to compare against")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--n-probs", type=int, default=50)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--sets", nargs="+", default=["jevbench:test:231", "sni:val:300", "reflex:val:150"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    lid = [tok(" " + L, add_special_tokens=False)["input_ids"][0] for L in LETTERS]
    srv = subprocess.Popen([a.server, "-m", a.gguf, "-t", str(a.threads), "-ngl", "0", "-c", "8192",
                            "--port", str(a.port), "--host", "127.0.0.1", "-np", "1"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(300):
            try:
                if json.loads(urllib.request.urlopen(base + "/health", timeout=2).read()).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise SystemExit("llama-server did not become healthy")
        report = {"gguf": a.gguf, "threads": a.threads, "n_probs": a.n_probs,
                  "cpu": next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
                  "loadavg_start": open("/proc/loadavg").read().split()[:3], "sets": {}}
        warm = 5
        for spec in a.sets:
            name, split, n = spec.split(":")
            rows = [r for r in json.load(open(ROOT / "data" / name / f"{split}.json"))
                    if 2 <= len(r["options"]) <= 26 and 0 <= r["answer_index"] < len(r["options"])][: int(n)]
            gpu = {}
            gp = ROOT / "results" / "evals" / a.gpu_tag / f"{name}.{split}.orig.jsonl"
            if a.gpu_tag and gp.exists():
                gpu = {j["i"]: j for j in map(json.loads, open(gp))}
            lat, ptoks, correct, missing, agree, compared, skipped_trunc = [], [], 0, 0, 0, 0, 0
            for i, r in enumerate(rows):
                t0 = time.perf_counter()
                res = post(base + "/completion", {"prompt": build_prompt(r["state"], r["question"], r["options"]),
                                                  "n_predict": 1, "n_probs": a.n_probs, "temperature": 0,
                                                  "cache_prompt": False})
                dt = time.perf_counter() - t0
                if i >= warm or len(rows) <= warm:
                    lat.append(dt)
                ptoks.append(res.get("tokens_evaluated") or (res.get("timings") or {}).get("prompt_n") or 0)
                top = res["completion_probabilities"][0]["top_logprobs"]
                got = {t["id"]: t["logprob"] for t in top}
                k = len(r["options"])
                if not all(lid[j] in got for j in range(k)):
                    missing += 1
                    continue
                lg = [got[lid[j]] for j in range(k)]
                pred = max(range(k), key=lg.__getitem__)
                correct += pred == r["answer_index"]
                if i in gpu:
                    # the GPU eval left-truncated this prompt (max_len); the CPU saw all of it,
                    # so a disagreement here would not be a CPU/quantization effect
                    if gpu[i].get("trunc"):
                        skipped_trunc += 1
                        continue
                    compared += 1
                    agree += pred == max(range(k), key=gpu[i]["logits"].__getitem__)
            n = len(rows)
            report["sets"][spec] = {
                "n": n, "missing_letters": missing,
                "acc_cpu_over_scored": correct / max(1, n - missing),
                "acc_cpu_missing_as_wrong": correct / max(1, n),
                "gpu_agreement": (agree / compared) if compared else None, "compared": compared,
                "excluded_gpu_truncated": skipped_trunc,
                "latency_s_p50": pct(lat, 50), "latency_s_p90": pct(lat, 90), "latency_s_mean": statistics.mean(lat),
                "prompt_tokens_p50": pct(ptoks, 50), "prompt_tokens_p90": pct(ptoks, 90)}
            s = report["sets"][spec]
            print(f"[{spec}] acc={s['acc_cpu_missing_as_wrong']:.3f} missing={missing} "
                  f"gpu_agree={s['gpu_agreement']} p50={s['latency_s_p50']*1000:.0f}ms "
                  f"p90={s['latency_s_p90']*1000:.0f}ms prompt_tok_p50={s['prompt_tokens_p50']}", flush=True)
        report["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
        json.dump(report, open(a.out, "w"), indent=1)
        print("SAVED", a.out)
    finally:
        srv.terminate()
        srv.wait(timeout=30)


if __name__ == "__main__":
    main()
