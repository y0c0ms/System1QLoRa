---
license: apache-2.0
library_name: peft
tags:
  - decision-model
  - system-one
  - calibration
  - lora
  - qlora
  - typesafe
base_model:
  - Qwen/Qwen3-0.6B
  - Qwen/Qwen3-4B-Instruct-2507
metrics:
  - accuracy
  - brier_score
  - expected_calibration_error
---

# System-One QLoRA

Two small, open **single-pass decision scorers** in the shape of TypeSafe's *Jev* ("System One"):
a `state` plus a multiple-choice `question` goes in, a **calibrated distribution over the options**
comes out in one forward pass — no autoregressive generation, no sampling, deterministic.

This is a research and fun weekend project: an independent reproduction of the *shape* of a
closed decision model, built on public benchmarks. The benchmark construction, the letter-logprob
scoring protocol, and the recorded pitfalls are the reusable parts — the adapters are one point
in that design space. **Improvements on top of this work are very welcome** — see
[the repository](https://github.com/y0c0ms/System1QLoRa).

| | base | trainable | latency / decision |
|---|---|---|---|
| [`06b/`](https://huggingface.co/yocoms/system1-qlora/tree/main/06b) | Qwen3-0.6B | LoRA r=16, α=32 | ~53 ms on GPU; ~0.4 s on CPU (fp32) |
| [`4b/`](https://huggingface.co/yocoms/system1-qlora/tree/main/4b) | Qwen3-4B-Instruct-2507 | LoRA r=16, α=32 | ~165 ms on GPU |

## Scoring mechanism

The prompt ends with `\n\nAnswer:`. We take the next-token log-probability of each option's
letter (` A`, ` B`, …), softmax over the present options, and apply a temperature fitted on a
validation split. The argmax is the decision; the softmax is the calibrated confidence.
The letter readout caps at 26 options.

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

subfolder = "06b"  # or "4b"
base = "Qwen/Qwen3-0.6B" if subfolder == "06b" else "Qwen/Qwen3-4B-Instruct-2507"

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "yocoms/system1-qlora", subfolder=subfolder)
model.eval()

def decide(state: str, question: str, options: list[str], temperature: float = 1.0):
    opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
    prompt = f"State:\n{state}\n\nQuestion:\n{question}\n\nOptions:\n{opts}\n\nAnswer:"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    lp = model(**ids).logits[0, -1].log_softmax(-1)
    letters = [tok(f" {chr(65+i)}", add_special_tokens=False)["input_ids"][0] for i in range(len(options))]
    probs = (lp[letters] / temperature).softmax(-1)
    return probs.argmax().item(), probs.max().item()

idx, confidence = decide(
    state="Backlight on, panel bright, no image but OSD menus display normally.",
    question="What is the most likely faulty component?",
    options=["T-con board", "Mainboard", "Backlight driver", "Power supply"],
)
```

Notes:

- Do **not** route the prompt through a chat template — a reasoning template injects tokens
  before the letter and the score reads the wrong position. Score through the raw forward pass.
- On CPU, load with `torch_dtype=torch.float32` (bf16 CPU matmuls are slow); the 0.6B fits
  comfortably in 16 GB RAM.

## Evaluation

Held-out test, calibrated (accuracy / ECE). The closed **Jev 1.13**
(`opencode-zen/jev-1.13`) is scored on identical items as a reference point, not as a rival.

| benchmark | 0.6B | 4B | Jev 1.13 |
|---|---|---|---|
| SNI (held-out predicates) | 0.613 / .037 | **0.707** / .050 | 0.838 |
| reflex (math + code) | 0.553 / .059 | 0.558 / .072 | 0.543 |
| BFCL (tool selection) | 0.885 / .032 | 0.920 / .034 | 0.957 |
| abstention ("none fits") | **0.960** / .019 | 0.924 / .035 | 0.740 |

A 0.6B model scoring at 96% abstention accuracy and matching a large prompted model on
tool selection was the surprise of the project. The SNI gap to Jev is knowledge/reasoning
capacity, not format — see below.

## Recipe

- Corpus: SNI + reflex train splits, with **absence-augmentation** (abstain/NOTA variants,
  upweighted absence class) and format-matched NLI. BFCL is held out of all training by policy —
  its score is pure zero-shot transfer.
- **The NLI fix**: SNI's held-out NLI tasks are *"pick which of 3 candidates is neutral"*, not
  single-pair label classification. Training standard MNLI left `mnli_neutral` at 0.16 (below
  chance). Rebuilding the data in the select-of-3 format lifted it **0.16 → 0.88** on the 0.6B.
  The format mismatch, not model capacity, was the bottleneck.
- QLoRA: 4-bit NF4, r=16, α=32, batch 4 × seq 384, ~1.25 epochs, completion-only loss.

Code, benchmark builders and the full lab journal:
[github.com/y0c0ms/System1QLoRa](https://github.com/y0c0ms/System1QLoRa)

## Limitations

- **Single-seed numbers on one hardware setup** — treat them as a worked example of the
  protocol, not a leaderboard entry.
- **Small test splits** (hundreds to ~2k rows): enough to separate signal from chance, not for
  tight confidence intervals.
- Letter readout caps at 26 options; prompts are truncated from the left at 1536 tokens.
- English-only benchmarks; generalization beyond this distribution is untested.
- Known weak spot: code-vulnerability detection sits near coin-flip in both adapters.

## License & attribution

Apache-2.0, like the Qwen base models. **Not affiliated with TypeSafe AI**; independent
reproduction of the *shape* of Jev. Benchmarks derive from
[Natural Instructions](https://github.com/allenai/natural-instructions),
[MATH](https://github.com/hendrycks/math), a public code-vulnerability dataset, and
[`llamastack/bfcl_v3`](https://huggingface.co/datasets/llamastack/bfcl_v3) — each keeps its
upstream license.

**Contributions welcome** — better calibration methods, more tasks, a bigger corpus, distillation
from a stronger scorer. If you build on this, a link back to the repo is appreciated.
