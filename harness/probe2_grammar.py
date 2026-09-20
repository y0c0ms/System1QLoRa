#!/usr/bin/env python3
"""T2 step 2: get a probability for EVERY option, not just the popular ones.

THE PROBLEM probe_logprobs.py found. Asking for the top-N tokens at the answer
position returns whatever the model likes most - which was 'A' (p=0.67) and then
'The', 'When', 'This'. Options B and C never appeared. Renormalising over "the
option letters we happened to see" silently drops options, and a calibration
number computed that way is measuring the wrong distribution.

Two candidate fixes, both tried here:

  1. RAISE top_logprobs. Cheap, but it only widens the window - a confident model
     can still push a rare option past any fixed N, and you cannot tell from the
     response whether that happened. It does not fix the class of bug.

  2. CONSTRAIN THE SAMPLER with a GBNF grammar so the only legal next token is an
     option letter. The returned distribution is then over exactly the options,
     with no prose competing for mass and nothing truncated away. This is the
     right shape: it is what "typed decision" means at the sampler level.

Reports the raw response structure too, because probe 1 guessed the native
endpoint's key names wrong and reported success on an empty list.
"""

import json
import math
import time
import urllib.error
import urllib.request

BASE = __import__("os").environ.get("LLAMA_SERVER_URL", "http://localhost:11435")
MODEL = "qwen3.6-35b"
LETTERS = "ABC"

PROMPT = """State:
My card was declined at the supermarket this morning and I have no idea why.

Question:
Which support queue should this go to?

Options:
A. billing
B. technical
C. sales

Answer:"""


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), (time.time() - t0) * 1000, None
    except urllib.error.HTTPError as e:
        return None, (time.time() - t0) * 1000, "HTTP %s: %s" % (
            e.code, e.read()[:400].decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001
        return None, (time.time() - t0) * 1000, "%s: %s" % (type(e).__name__, e)


def show(tag, top):
    """Print the letter mass and, crucially, which letters are MISSING."""
    seen = {}
    for e in top:
        t = e["token"].strip()
        if t in LETTERS and t not in seen:
            seen[t] = math.exp(e["logprob"])
    missing = [c for c in LETTERS if c not in seen]
    total = sum(seen.values())
    print("  %s" % tag)
    print("    letters found : %s" % (sorted(seen) or "NONE"))
    print("    letters MISSING: %s" % (missing or "none"))
    print("    mass on letters: %.4f  (the rest went to prose)" % total)
    if total > 0:
        for c in LETTERS:
            if c in seen:
                print("      p(%s) = %.4f   renormalised %.4f" % (c, seen[c], seen[c] / total))
    return not missing


def wide_window(n):
    body, ms, err = post("/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1, "temperature": 1.0,
        "logprobs": True, "top_logprobs": n,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    if err:
        print("  top_logprobs=%d FAILED %s" % (n, err))
        return False
    top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    print("  [%.0f ms, %d tokens returned]" % (ms, len(top)))
    return show("top_logprobs=%d" % n, top)


def grammar():
    """Restrict the next token to an option letter, then read the distribution."""
    gbnf = 'root ::= [%s]' % LETTERS
    body, ms, err = post("/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1, "temperature": 1.0,
        "logprobs": True, "top_logprobs": 20,
        "grammar": gbnf,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    if err:
        print("  grammar FAILED %s" % err)
        return False
    top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    print("  [%.0f ms, %d tokens returned]" % (ms, len(top)))
    print("    chose: %r" % body["choices"][0]["message"]["content"])
    return show("grammar-constrained", top)


if __name__ == "__main__":
    print("=== fix 1: just ask for a wider window ===")
    a = wide_window(20)
    b = wide_window(60)
    print("\n=== fix 2: constrain the sampler with a GBNF grammar ===")
    c = grammar()
    print("\n=== verdict ===")
    print("  top_logprobs=20      covers all options: %s" % a)
    print("  top_logprobs=60      covers all options: %s" % b)
    print("  grammar-constrained  covers all options: %s" % c)
