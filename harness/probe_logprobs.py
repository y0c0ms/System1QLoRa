#!/usr/bin/env python3
"""T2 step 1: can we get per-token probabilities out of llama-swap at all?

WHY THIS EXISTS BEFORE THE BENCHMARK. The whole zero-install baseline rests on
reading the probability the model assigns to each option letter at ONE position.
If the server will not surface those numbers, the baseline is impossible and the
project goes straight to PyTorch - so this is checked first, with nothing else
built on top of it.

Two routes are tried because they fail differently:

  /v1/chat/completions  + logprobs/top_logprobs
      OpenAI-compatible, but it applies the model's CHAT TEMPLATE, which wraps
      the prompt in role markers and can move or retokenise the answer position.

  /completion (llama.cpp native, via llama-swap's /upstream/<model>/ path)
      Raw prompt, no template. `n_probs` asks for the top-N tokens at each
      generated position. This is the one we want if it works.

Nothing here is a measurement. It reports which route returns usable numbers.
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = __import__("os").environ.get("LLAMA_SERVER_URL", "http://localhost:11435")
MODEL = "qwen3.6-35b"

# A deliberately easy question: if the model cannot get THIS right, the problem
# is the plumbing, not the model.
PROMPT = """State:
My card was declined at the supermarket this morning and I have no idea why.

Question:
Which support queue should this go to?

Options:
A. billing
B. technical
C. sales

Answer:"""


def post(path: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), (time.time() - t0) * 1000, None
    except urllib.error.HTTPError as e:
        return None, (time.time() - t0) * 1000, "HTTP %s: %s" % (e.code, e.read()[:300].decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001 - probe reports, never raises
        return None, (time.time() - t0) * 1000, "%s: %s" % (type(e).__name__, e)


def try_chat():
    print("\n--- route 1: /v1/chat/completions with top_logprobs ---")
    body, ms, err = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": 20,
        # Qwen3.6 is a thinking model; a reasoning preamble would put the answer
        # nowhere near token 0.
        "chat_template_kwargs": {"enable_thinking": False},
    })
    print("  %.0f ms" % ms)
    if err:
        print("  FAILED", err)
        return False
    try:
        lp = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        print("  no logprobs in response. keys:", list(body.get("choices", [{}])[0].keys()))
        return False
    print("  top tokens at position 0:")
    for e in lp[:12]:
        print("    %-14r logprob=%8.4f  p=%.4f" % (e["token"], e["logprob"], 2.718281828 ** e["logprob"]))
    return True


def try_native():
    print("\n--- route 2: /upstream/%s/completion with n_probs ---" % MODEL)
    body, ms, err = post("/upstream/%s/completion" % MODEL, {
        "prompt": PROMPT,
        "n_predict": 1,
        "n_probs": 40,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "cache_prompt": True,
    })
    print("  %.0f ms" % ms)
    if err:
        print("  FAILED", err)
        return False
    probs = body.get("completion_probabilities")
    if not probs:
        print("  no completion_probabilities. top-level keys:", sorted(body)[:20])
        return False
    top = probs[0].get("top_probs") or probs[0].get("probs") or []
    print("  top tokens at position 0:")
    for e in top[:12]:
        tok = e.get("tok_str", e.get("token"))
        p = e.get("prob", e.get("p"))
        print("    %-14r p=%s" % (tok, p))
    return True


if __name__ == "__main__":
    print("probing %s on %s" % (MODEL, BASE))
    print("NOTE: the model is unloaded; the first call pays the load cost (22.4 GB GGUF).")
    ok_native = try_native()
    ok_chat = try_chat()
    print("\n=== verdict ===")
    print("  native /completion n_probs : %s" % ("USABLE" if ok_native else "no"))
    print("  /v1 chat top_logprobs      : %s" % ("USABLE" if ok_chat else "no"))
    sys.exit(0 if (ok_native or ok_chat) else 1)
