#!/usr/bin/env python3
"""Tiny stdlib HTTP server exposing the trained 4B decision scorer.

Loads base + LoRA once, then answers:
  GET  /health                      -> {"ok": true, ...}
  POST /score {state,question,options[]} -> {options, probs, pick, latency_ms}

Runs INSIDE the rocm container (needs the GPU). No third-party web deps.
"""

import json
import string
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from eval_decider import text_field  # canonical field rendering; see eval_decider.text_field

BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "results/jevlike4b_lora"
LETTERS = string.ascii_uppercase
PORT = 8900

print("loading tokenizer/model...", flush=True)
TOK = AutoTokenizer.from_pretrained(BASE)
TOK.truncation_side = "left"
_bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_use_double_quant=True,
                          bnb_4bit_compute_dtype=torch.bfloat16)
MODEL = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=_bnb, torch_dtype=torch.bfloat16,
    device_map={"": 0}, attn_implementation="eager")
MODEL = PeftModel.from_pretrained(MODEL, ADAPTER)
MODEL.eval()
LID = {L: TOK(" " + L, add_special_tokens=False)["input_ids"][0] for L in LETTERS}
print("model ready", flush=True)


def build_prompt(state, question, options):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(options))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (text_field(state), text_field(question), opts))


@torch.no_grad()
def score(state, question, options):
    t0 = time.time()
    ids = TOK(build_prompt(state, question, options), return_tensors="pt",
              truncation=True, max_length=1536)
    ids = {k: v.to(MODEL.device) for k, v in ids.items()}
    lg = MODEL(**ids).logits[0, -1].float()
    raw = torch.tensor([lg[LID[LETTERS[i]]] for i in range(len(options))])
    probs = torch.softmax(raw, dim=-1).tolist()
    pick = max(range(len(probs)), key=lambda i: probs[i])
    return {"options": options, "probs": probs, "pick": pick,
            "latency_ms": round((time.time() - t0) * 1000, 1)}


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {})

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "base": BASE, "adapter": ADAPTER})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/score":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            opts = [str(o) for o in req.get("options", [])]
            if not (2 <= len(opts) <= 26):
                return self._send(400, {"error": "need 2-26 options"})
            out = score(str(req.get("state", "")), str(req.get("question", "")), opts)
            self._send(200, out)
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"serving on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
