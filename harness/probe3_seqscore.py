#!/usr/bin/env python3
"""Can we get logP(option_text | prompt) out of llama.cpp?

WHY WE NEED THIS. Letter scoring reads one token and asks which of A/B/C is most
likely there. It failed three ways: options past Z have no letter (230 test rows
dropped), letters fall outside top_n and get a sentinel (26.5% of ag_news rows),
and after "Answer:" a raw continuation wants prose - it wants to write "World",
not "A" - so the letters sink out of the window entirely.

Sequence scoring asks the question directly: how likely is THIS OPTION TEXT as
the continuation? No window, no alphabet, no letter-versus-word conflict.

THE MECHANISM. llama.cpp has no endpoint for prompt-token logprobs, so we force
the exact option text with a GBNF grammar and read the logprob the model assigned
to each forced token. probe2 established that llama.cpp reports the PRE-grammar
distribution - which is exactly what makes this valid: the grammar controls which
tokens are emitted, not what probability is reported for them.

Also checks `cache_prompt`, which decides whether this is affordable: every option
of a row shares the same prefix, so the prefill should be paid once per row rather
than once per option.
"""

import json
import time
import urllib.request

BASE = __import__("os").environ.get("LLAMA_SERVER_URL", "http://localhost:11435") + "/upstream/qwen3.6-35b"

PROMPT = ("State:\nThe Dow Jones industrial average fell 200 points as investors "
          "reacted to weak earnings from several major banks.\n\n"
          "Question:\nWhat topic is this news about?\n\nAnswer:")
OPTIONS = ["World", "Sports", "Business", "Sci/Tech"]


def gbnf_literal(s):
    """A grammar that admits exactly one string."""
    esc = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return 'root ::= "%s"' % esc


def score(option, cache=True):
    body = {
        "prompt": PROMPT,
        # Enough tokens for the option; the grammar stops it at the right place.
        "n_predict": 32,
        "n_probs": 1,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0,
        "grammar": gbnf_literal(" " + option),
        "cache_prompt": cache,
    }
    req = urllib.request.Request(BASE + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    ms = (time.time() - t0) * 1000
    cp = d.get("completion_probabilities") or []
    toks = [e.get("token") for e in cp]
    lps = [e.get("logprob") for e in cp if e.get("logprob") is not None]
    t = d.get("timings", {}) or {}
    return {
        "content": d.get("content"),
        "tokens": toks,
        "n_tok": len(lps),
        "total_logprob": sum(lps) if lps else None,
        "mean_logprob": (sum(lps) / len(lps)) if lps else None,
        "ms": ms,
        "prompt_ms": t.get("prompt_ms"),
        "cached": d.get("tokens_cached"),
    }


if __name__ == "__main__":
    print("  first call (cold prefix):")
    r = score(OPTIONS[0])
    print("    emitted %r as %d tokens %s" % (r["content"], r["n_tok"], r["tokens"]))
    print("    total=%.4f mean=%.4f | %.0f ms (prompt %.0f ms) cached=%s"
          % (r["total_logprob"], r["mean_logprob"], r["ms"], r["prompt_ms"] or 0, r["cached"]))
    print()
    print("  remaining options (prefix should now be cached):")
    rows = [("  %-10s" % OPTIONS[0], r)]
    for o in OPTIONS[1:]:
        rr = score(o)
        rows.append(("  %-10s" % o, rr))
        print("    %-10s %d tok  total=%8.4f  mean=%7.4f  %5.0f ms (prompt %4.0f ms)"
              % (o, rr["n_tok"], rr["total_logprob"], rr["mean_logprob"], rr["ms"],
                 rr["prompt_ms"] or 0))
    print()
    best_t = max(rows, key=lambda x: x[1]["total_logprob"])
    best_m = max(rows, key=lambda x: x[1]["mean_logprob"])
    print("  argmax by TOTAL logprob : %s" % best_t[0].strip())
    print("  argmax by MEAN  logprob : %s   (length-normalised)" % best_m[0].strip())
    print("  truth                   : Business")
