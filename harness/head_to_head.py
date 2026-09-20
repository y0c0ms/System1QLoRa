#!/usr/bin/env python3
"""Qwen3.6-35B-A3B vs Qwen3.8-27B-IQ4_XS on the same question, same box.

WHY TOK/S IS THE WRONG HEADLINE. 3.6 is served thinking-OFF; 3.8 cannot disable
reasoning at all. A model at the same tok/s that emits 200 reasoning tokens
before its answer is far slower to a USABLE reply. So the number reported here
is time-to-finished-answer, with tok/s and reasoning-token count alongside it to
show where the time went.

Run against one endpoint at a time - they are exclusive on a 16 GiB card.
"""

import json
import sys
import time
import urllib.request


def ask(base, model, question, max_tokens=512):
    body = {"model": model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens, "temperature": 0.7, "stream": False}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    wall = time.time() - t0
    msg = d["choices"][0]["message"]
    t = d.get("timings", {}) or {}
    u = d.get("usage", {}) or {}
    rc = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    return {
        "wall_s": wall,
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "decode_tok_s": t.get("predicted_per_second"),
        "prefill_tok_s": t.get("prompt_per_second"),
        "reasoning_chars": len(rc),
        "answer_chars": len(content),
        "answer": content.strip()[:220],
    }


QUESTIONS = [
    ("short factual", "What is the capital of Portugal? Answer in one sentence."),
    ("typical ask", "Explain in one short paragraph why a Kubernetes liveness probe "
                    "can restart a healthy-but-slow container."),
]

if __name__ == "__main__":
    base, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    out = {}
    for name, q in QUESTIONS:
        print("  [%s] %s ..." % (tag, name), flush=True)
        r = ask(base, model, q)
        out[name] = r
        print("    wall %.1fs | decode %.1f tok/s | prefill %.1f tok/s | "
              "completion %s tok | reasoning %d chars"
              % (r["wall_s"], r["decode_tok_s"] or 0, r["prefill_tok_s"] or 0,
                 r["completion_tokens"], r["reasoning_chars"]), flush=True)
        print("    answer: %s" % r["answer"][:150], flush=True)
    json.dump(out, open("results/h2h_%s.json" % tag, "w"), indent=1)
