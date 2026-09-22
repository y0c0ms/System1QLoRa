#!/usr/bin/env python3
"""Smoke-test the 0.6B System-One adapter on CPU.

Loads Qwen3-0.6B + yocoms/system1-qlora:06b, scores the 10 hard cases from
results/demo_hard_results.json with the same letter-logprob protocol as
hf_score.py, and compares adapter vs unmodified base to verify the LoRA
actually changes decisions.
"""

import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = "yocoms/system1-qlora"
SUBFOLDER = "06b"
_local = ROOT / "models" / "Qwen3-0.6B" / "model.safetensors"
BASE = str(_local.parent) if _local.exists() else "Qwen/Qwen3-0.6B"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

torch.set_num_threads(max(1, torch.get_num_threads() - 2))


def _adapter_snapshot_path(subfolder):
    local = ROOT / "models" / ("system1-qlora-" + subfolder)
    if (local / "adapter_model.safetensors").exists():
        return str(local)
    from huggingface_hub import snapshot_download
    return str(Path(snapshot_download(ADAPTER, allow_patterns=[subfolder + "/*"],
                                      local_files_only=True)) / subfolder)


def build_prompt(row):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(row["options"]))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (row["state"], row["question"], opts))


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.truncation_side = "left"
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager")
    print(f"base loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    letter_id = {L: tok(" " + L, add_special_tokens=False)["input_ids"][0]
                 for L in LETTERS}

    @torch.no_grad()
    def score(m, row):
        ids = tok(build_prompt(row), add_special_tokens=True,
                  return_tensors="pt", truncation=True, max_length=1536)
        t = time.perf_counter()
        logits = m(**ids).logits[0, -1]
        dt = (time.perf_counter() - t) * 1000
        lp = torch.log_softmax(logits.float(), dim=-1)
        return [lp[letter_id[LETTERS[i]]].item() for i in range(len(row["options"]))], dt

    rows = json.load(open(ROOT / "results" / "demo_hard_results.json"))["cases"]

    # base (no adapter) for comparison
    base_res = []
    for r in rows:
        lg, dt = score(model, r)
        base_res.append((lg, dt))

    t0 = time.perf_counter()
    model = PeftModel.from_pretrained(model, _adapter_snapshot_path(SUBFOLDER))
    model.eval()
    print(f"adapter loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    n_correct = n_base_correct = 0
    lats = []
    out_cases = []
    for r, (blg, bdt) in zip(rows, base_res):
        lg, dt = score(model, r)
        lats.append(dt)
        pick = max(range(len(lg)), key=lambda i: lg[i])
        bpick = max(range(len(blg)), key=lambda i: blg[i])
        mx = max(lg)
        probs = [2.718281828 ** (x - mx) for x in lg]
        s = sum(probs)
        probs = [p / s for p in probs]
        ok = pick == r["expect"]
        bok = bpick == r["expect"]
        n_correct += ok
        n_base_correct += bok
        agree = "same" if bpick == pick else "DIFF"
        print(f"[{'ok' if ok else 'MISS'}] {r['tag']:38s} pick={pick} conf={probs[pick]:.3f} "
              f"base={bpick}({bok}) {agree} {dt:.0f}ms", flush=True)
        out_cases.append({"tag": r["tag"], "pick": pick, "expect": r["expect"],
                          "correct": ok, "probs": probs, "latency_ms": dt,
                          "base_pick": bpick, "base_correct": bok})

    out = {"base": BASE, "adapter": ADAPTER, "subfolder": SUBFOLDER, "device": "cpu",
           "n_total": len(rows), "n_correct_adapter": n_correct,
           "n_correct_base": n_base_correct,
           "latency_ms_mean": sum(lats) / len(lats), "cases": out_cases}
    dest = ROOT / "results" / "test_06b_cpu.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nADAPTER {n_correct}/{len(rows)}  BASE {n_base_correct}/{len(rows)}  "
          f"mean {sum(lats)/len(lats):.0f} ms/decision (CPU fp32)")
    print("SAVED", dest)


if __name__ == "__main__":
    main()
