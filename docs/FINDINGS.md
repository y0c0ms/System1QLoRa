# Findings

Running log. Every entry is something measured on this hardware, with the command that
produced it. Claims without a measurement go in the map's fog, not here.

Hardware: yocomsmain — Ryzen 7600X, 32 GB, RX 7900 GRE (gfx1100, 16 GiB), Fedora 44.

---

## 2026-09-18 — T1: Python 3.14 is not a blocker

`pip install --dry-run "transformers>=4.49.0" "peft>=0.20.0" datasets` on the system 3.14.7
resolves cleanly:

| package | resolved |
|---|---|
| transformers | 5.17.0 |
| peft | **0.21.0** |
| datasets | 5.0.1 |
| torch | 2.14.0 |
| accelerate | 1.15.0 |
| tokenizers / safetensors / numpy | 0.23.2 / 0.8.0 / 2.5.3 |

`peft 0.21.0` clears the ~0.20 floor that `pretrained-scorer/adapter_config.json`
(`peft_version: 0.20.0`) requires, so **no Python 3.12 container is needed**.

**Trap, recorded before someone falls in it:** the default PyPI `torch` pulled the entire NVIDIA
stack — `nvidia-cublas`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-cusparselt`, `triton`,
`cuda-toolkit 13.0.3`. Several GB of libraries that cannot run on an AMD card, and
`torch.cuda.is_available()` would be **False** afterwards. Any real install must pass an explicit
`--index-url` (`.../whl/cpu` or `.../whl/rocmX.Y`), or use Fedora's `python3-torch`, which is a
HIP build.

**Open:** whether ROCm wheels exist for cp314 is *not* answered by this. Unverified.

---

## 2026-09-18 — T2: probabilities can be read out of llama-swap, with one real limit

`harness/probe_logprobs.py`, `harness/probe2_grammar.py` against `qwen3.6-35b` on
`http://localhost:11435`.

### What works

`/v1/chat/completions` with `logprobs: true`, `top_logprobs: N`, `max_tokens: 1` and
`chat_template_kwargs: {enable_thinking: false}` returns a usable distribution at the answer
position. **No torch, no training, GPU already serving.**

### Latency, first real numbers

| | ms |
|---|---|
| cold call (22.4 GB GGUF load) | **27,107** |
| warm call, ~80-token prompt, 1 token out | **101 – 114** |

Three warm samples: 114, 101, 102 ms. Not a benchmark — one prompt, no sweep, n=3 — but it puts
a 35B MoE at ~100 ms per decision on this box, against Jev's claimed 70–500 ms and
system-one-gemma's unmeasured "~50 ms".

### The limit that shapes the design

**`top_logprobs` truncates, silently.** With 3 options:

| window | letters found | missing |
|---|---|---|
| `top_logprobs=20` | A, B | **C** |
| `top_logprobs=60` | A, B, C | none |

Only **67%** of the probability mass sat on option letters at all; the rest went to prose
continuations (`'The'` 0.216, `'When'` 0.044, `'This'` 0.032). Renormalising over the letters
that happened to appear would have silently dropped C and reported a clean-looking distribution.

**So any option must be counted as missing and reported, never renormalised away.** The probes
print `letters MISSING` for exactly this reason.

### GBNF grammar does NOT solve it — approach eliminated

Constraining the sampler with `root ::= [ABC]` **was accepted** (the model emitted `A`), but the
returned `top_logprobs` were byte-identical to the unconstrained run: still 0.6701 total letter
mass, still no C. **llama.cpp reports the pre-grammar distribution.** The grammar changes what is
sampled, not what is measured. Do not reach for it again to fix a truncation problem.

### Overconfidence is visible immediately

Raw `p(A) = 0.6700`; renormalised over letters, `p(A) = 0.9998`. That near-certainty on a
genuinely ambiguous support-queue question is exactly the miscalibration temperature scaling
exists to correct — upstream fitted `T = 2.35`, which says the same thing about the 270M model.

### Still open

- Options beyond 26 have no letter (banking77 has 77). Upstream skips these and counts them;
  we should do the same and report the count, not hide it.
- The native `/completion` route returned `completion_probabilities` but no token list under the
  keys guessed in probe 1 — **and probe 1's verdict line wrongly reported it USABLE**. The chat
  route is what actually works. Structure of the native response is still unknown.

---

## 2026-09-18 — corpus shape confirmed, and the cap that selects one task

`harness/fetch_data.py`, revision pinned to `fe08107`.

| split | rows | letter-scoreable | skipped (>26 options) | distinct questions |
|---|---|---|---|---|
| val | 1,452 | 1,292 | 160 | **36** |
| test | 1,751 | 1,521 | 230 | **36** |

36 distinct questions in **both** splits — the same 36 strings, confirming the split is by example
and not by predicate. `banking77` (77 options) and `tickets_queue` (52) carry every skipped row.

**Bug worth recording.** `--limit 40` was implemented as a global slice. The corpus is ordered by
task and banking77 is first, so it selected 40 rows that *all* have 77 options, scored **zero**,
and then divided by zero computing accuracy. A global cap does not sample this benchmark, it picks
one task. Fixed by `take_per_task`. Upstream's `take_per_task` carries a comment about the same
trap — the warning was there and the mistake still got made.

---

## 2026-09-18 — T4 unblocked without an HF token

The obvious blocker: `google/gemma-3-270m` is **`gated=manual`**. No token is configured on this
machine, and `hf_hub_download` returns `GatedRepoError: 401`. Accepting the licence is a human
action.

**Workaround, verified:** `mlx-community/gemma-3-270m-bf16` is an ungated re-upload of the **base**
model — `Gemma3ForCausalLM`, `model_type: gemma3_text`, **`hidden_size: 640`**, `quantization:
none`, standard HF safetensors. 640 is exactly the width of the shipped adapter's
`base_model.model.score.weight` `[1, 640]`.

Loading `Gemma3TextForSequenceClassification(num_labels=1)` on it and applying
`upstream/pretrained-scorer` works:

```
base score head shape : (1, 640)
head changed by adapter: True      <- the TRAINED head landed, not a random init
head shape after       : (1, 640)  dtype=torch.float32
```

`lm_head.weight UNEXPECTED / score.weight MISSING` in the load report is the expected signature of
converting a CausalLM checkpoint to sequence classification.

**Caveat that travels with any number produced this way:** these are re-uploaded weights, not
byte-verified against Google's original. Use the `-it` variant by mistake and it would be a
*different base* from the one the adapter was trained on.

**Environment for the torch arm** (`.venv`, explicit `--index-url .../whl/cpu`):
`torch 2.14.0+cpu` (`cuda=False`, no NVIDIA packages), `transformers 5.17.0`, `peft 0.21.0`,
`datasets 5.0.1`. `Gemma3TextForSequenceClassification` still exists in transformers 5.x, so
upstream's by-name import survives the major-version jump.

---

## 2026-09-18 — T4: the honest test eval. **The released scorer is BETTER than advertised.**

`system_one.py eval --base-model mlx-community/gemma-3-270m-bf16 --model-dir
upstream/pretrained-scorer --split test`. Full test split, no per-task cap.
`temperature 2.6 fitted on val (val_nll 0.76)`, applied to test — never fitted on the rows scored.

| | n | accuracy | ECE | Brier |
|---|---|---|---|---|
| **Honest, full test split** | **1751** | **0.6796** | **0.0220** | **0.4031** |
| Published (val, ≤576 q, T in-sample) | ≤576 | 0.6443 | 0.0471 | 0.4542 |

Per task:

| task | n | acc | ECE |
|---|---|---|---|
| ag_news | 200 | 0.9100 | 0.0674 |
| banking77 | 150 | 0.6600 | 0.0455 |
| go_emotions | 641 | 0.8393 | 0.0194 |
| mmlu | 250 | **0.3040** | 0.0878 |
| tickets_language | 80 | 0.9125 | 0.0096 |
| tickets_priority | 80 | 0.4125 | 0.1137 |
| tickets_queue | 80 | **0.2625** | 0.0972 |
| tickets_type | 70 | 0.6714 | 0.1287 |
| yelp_score | 200 | 0.6050 | 0.1606 |

### I predicted this would go the other way, and it did not

The audit established — correctly, and it is still true — that the published numbers are
**val-split, ≤576 questions, with the temperature fitted on the very rows being scored**. From
that I concluded the published ECE was "optimistic by construction" and should be treated as
flattering. **Empirically it is the opposite: the honest number is better on all three metrics.**

The reason is the per-task cap, not the temperature. `EVAL_LIMIT = 64` gives every task equal
weight, so `mmlu` (0.304) and `tickets_queue` (0.263) count as much as `go_emotions` (0.839). In
the real test split `go_emotions` is **641 of 1751 rows — 37%** — so the honest average is pulled
up by the task mix. The in-sample temperature fit remains a genuine methodological flaw; it was
simply outweighed by the sampling effect running the other way.

**The lesson is about the claim, not the model.** "This metric was computed unsoundly" licenses
"do not trust this number". It does not license "the true number is worse". Re-running is what
settles direction, and the re-run is cheap.

**0.6796 / 0.0220 / 0.4031 is the baseline the logprob arm must beat.** Not 0.6443 / 0.0471.

---

## 2026-09-18 — T2 RESULT: the fine-tune wins, but not for the reason you would guess

Full clean run, quiet machine, checkpointed. `qwen3.6-35b` via llama-swap, `top_logprobs=100`,
temperature fitted on val (**T = 3.85**) and applied to test.

### The comparison must be on the same rows

The 270M eval scored all **1751** test rows. The logprob baseline can only score **1521** —
`banking77` (77 options) and `tickets_queue` (52) have no letter. Comparing the headline 0.6796
against 0.6443 would be comparing different row sets. Restricted to the 1521 rows both scored:

| | accuracy on the same 1521 rows |
|---|---|
| **270M fine-tuned scorer** | **0.7035** (1070 correct) |
| 35B logprob baseline | 0.6443 (980 correct) |
| | **+0.0592 for the 270M** |

**A 270M task-tuned scorer beats logprob scoring from a model ~130x its size.** The fine-tune is
not moot. The destination does not shrink.

### But the aggregate hides two opposite failures

| task | n | 270M | 35B logprob | delta |
|---|---|---|---|---|
| ag_news | 200 | 0.9100 | 0.8900 | -0.020 |
| **go_emotions** | 641 | **0.8393** | 0.5881 | **-0.251** |
| **mmlu** | 250 | 0.3040 | **0.6920** | **+0.388** |
| tickets_language | 80 | 0.9125 | 0.9125 | 0.000 |
| **tickets_priority** | 80 | **0.4125** | **0.0875** | **-0.325** |
| tickets_type | 70 | 0.6714 | 0.6714 | 0.000 |
| yelp_score | 200 | 0.6050 | 0.6250 | +0.020 |

**The big model buys world knowledge; the fine-tune buys task convention.** MMLU is knowledge a
270M model does not contain, and the 35B more than doubles it. go_emotions is a labelling
convention, and the fine-tune more than makes it back.

**`tickets_priority` is the sharpest result in the run.** The 35B scores **0.0875 on a 5-option
task whose random floor is 0.2000.** Below chance is not ignorance — a model that knew nothing
would score 0.20. It is systematically choosing wrong, which means it has the *concept* of
priority and not this dataset's *convention* for it (ordering, or what "low" vs "1" means). That
is precisely what a fine-tune supplies and a prompt does not.

### Calibration

| | temperature | ECE before | ECE after |
|---|---|---|---|
| 270M scorer (1751 rows) | 2.6 | — | **0.0220** |
| 35B logprob (1521 rows) | **3.85** | 0.1905 | 0.0848 |

The bigger model is *more* overconfident (T = 3.85 vs 2.6). **Caveat: these ECEs are over
different row sets and are not strictly comparable** — the 270M eval would need re-running
restricted to the 1521 rows to settle it. ECE does not decompose from per-task values, so it
cannot be recovered from the table above.

**A single global temperature is the wrong instrument.** It fixed the aggregate but *wrecked* a
task that was already fine: `ag_news` ECE went **0.0563 -> 0.3552** after calibration. Per-task or
input-conditional temperature is the obvious next question.

### Truncation is real at scale

**50 of 1521 rows (3.3%)** still had an option missing at `top_logprobs=100`. Those rows were
scored over an incomplete option set. Recorded, not renormalised away — AGENTS.md rule 2.

### Latency — throughput, not per-decision

`p50 1977 ms, p90 4078 ms, p99 5377 ms, max 6723 ms` under **4-way concurrency**; 1521 rows in
911 s = **1.7 decisions/sec**. This is not comparable to Jev's 70-500 ms until measured
single-stream.

Throughput decayed steadily within the run: **157 rows/min at the start, 99.8 at the end.** Cause
not established — longer states later in the corpus and KV-cache pressure are both candidates.
Noted, not explained.

---

## 2026-09-19 — Layer-wise probing: the head is reading the wrong layer

Frozen `gemma-3-270m`, no LoRA. One linear scorer per layer over the last-non-pad hidden state,
fitted on 2700 train rows (300/task, options capped at 8), temperature on 421 val rows, reported
on the **full 1751-row test split** — upstream's protocol, upstream's `encode`.

### Baselines first, because 0.51 means nothing without them

| | test accuracy | vs chance |
|---|---|---|
| random (mean 1/k, computed not assumed) | 0.3141 | — |
| majority class per task | 0.3946 | +0.08 |
| **layer 0 (token embeddings)** | **0.2901** | **−0.02 — below chance** |
| **best frozen layer (8 of 18)** | **0.5066** | **+0.1925** |
| **LoRA fine-tuned** | **0.6796** | +0.3655 |

Option counts run 2–77 and go_emotions (2 options) is 37% of the split, so the floor is 0.3141,
not 0.25 or 0.5.

### Two results

**1. A linear read of frozen features covers 53% of the distance from chance to the fine-tune.**
Not most of it. The fine-tune adds **+0.1730 accuracy** over the best frozen layer and is doing
real work — it is not merely reading off a representation that was already there. That settles
the question this experiment was run to answer: **the base representation is not sufficient**, so
a larger corpus is worth building.

**2. The signal peaks in the MIDDLE of the network and decays toward the output.**

```
layer  0 →  0.2901   (embeddings, below chance)
layer  1 →  0.4717   (one block buys almost everything)
layer  6 →  0.5060
layer  8 →  0.5066   ← peak
layer 13 →  0.4466
layer 18 →  0.4569   (final layer, where the head actually attaches)
```

**The scoring head is attached to layer 18. The best decision-relevant representation is at layer
8, and layer 18 is 0.0497 worse.** The last layer of a causal LM is specialised for next-token
prediction, not for classification — this is the measurable cost of that.

That is a concrete, cheap follow-up: attach the head at layer 8, or learn a weighted combination
over layers, and re-run the LoRA fit. `Gemma3TextForSequenceClassification` reads the final
hidden state, so this means a custom head, not a flag.

### A measurement bug this run exposed, and it was mine

Eight layers (9–16) first reported `T = 0.25` — **the exact floor of the temperature grid** — with
ECE ≈ 0.40, which read as a genuine mid-network calibration collapse. It was not. The search had
saturated: those layers wanted to sharpen *further* than the grid allowed. I added the saturation
detector after the earlier smoke test and then set the floor to upstream's 0.25 without asking
whether it suited this experiment.

Re-fitting on a geometric grid (0.02 → 20) removed the anomaly entirely — those layers now sit at
`T` 1.82–2.48 with ECE 0.049–0.084, in line with every other layer. **Accuracy was unaffected;
only the ECE column had been wrong.** Upstream's `fit_temperature` has the same 0.25 floor and
reports nothing when it saturates.

### fp16 overflow in the feature cache — real, small, fixed

`RuntimeWarning: overflow encountered in cast`. Gemma's large-magnitude outlier features exceed
float16's 65504 ceiling: 2–24 `inf` values per layer out of 13.2M, in layers 9–16 only, affecting
0.0–0.1% of rows. Train was clean, so the standardisation statistics were never contaminated.
Values are now clipped to the finite fp16 max rather than dropped — dropping the rows would have
silently shrunk the test set.

It is worth noting the overflow and the temperature saturation appeared in *the same layer band*,
which made one look like evidence for the other. They were unrelated.

---

## 2026-09-19 — Qwen3.8-27B IQ4_XS: runs, but only through the raw endpoint

`Qwen3.8-27B-UD-IQ4_XS.gguf`, 14,252,845,984 bytes, GGUF v3, 866 tensors — verified, not assumed.
Served with `/opt/llama/bin/llama-server` build **10050** (the August `~/llama/b10595` build is
**gone**), `--device Vulkan0 --n-gpu-layers 999 --ctx-size 4096 --parallel 1 --cache-type-k q8_0
--cache-type-v q8_0 --flash-attn on`.

### The August note skipped a quant tier

It recorded `UD-Q4_K_M` 16.46 GB and `UD-Q4_K_XL` 17.56 GB as too big and `UD-Q3_K_XL` 13.15 GB as
"what fits" — all correct — but never checked between them. **`UD-IQ4_XS` is 14.25 GB and fits.**

### Measured

| | |
|---|---|
| decode | **26.25 tok/s** (August measured 17.9–18.9 at Q3_K_XL) |
| prefill | **181.5 tok/s** — 215-token prompt in 1,185 ms |
| VRAM, `--parallel 4` (default) | 15.64 GiB of 15.98 — **98%** |
| VRAM, `--parallel 1` | **15.18 GiB** of 15.98 — 0.80 GiB free |
| load to healthy | ~10 s |

Faster than August on a *larger* quant, because context here is 4k rather than 32k. `--parallel 1`
matters: the default four slots quadruple the KV allocation for no benefit to a scorer.

**Prefill is the number that counts for scoring** — a scorer sends a few hundred tokens and
generates one. Decode tok/s is nearly irrelevant.

### The chat endpoint CANNOT score this model

`--chat-template-kwargs '{"enable_thinking":false}'` does not exist for Qwen3.8; reasoning cannot
be disabled. On a real scoring prompt through `/v1/chat/completions` the first generated token was
**`'We'`** and **no option letter appeared in the top 20**. Letter-logprob scoring reads position
0, so the method simply does not apply there.

**The reasoning is injected by the CHAT TEMPLATE.** The raw `/completion` endpoint applies no
template, so nothing is injected:

```
POST /completion  {"prompt": ..., "n_predict": 1, "n_probs": 40, "top_k": 0}
  content='\n\n'   665 ms
  '\n\n' -0.137 | '\n' -2.446 | ' A' -3.862 | ' B' -5.392 | ' The' -5.990 ...
  option letters missing: none - ALL THREE PRESENT
```

### And this resolves probe 1's false positive

The native response structure is `completion_probabilities[0]` with keys
`{bytes, id, logprob, token, top_logprobs}` — the list lives under **`top_logprobs`**.
`harness/probe_logprobs.py` guessed `top_probs` / `probs`, got an empty list, and reported the
route "USABLE" anyway because it tested that the *container* existed. Two days later the same
guess was still unexplained. Note the letters carry a **leading space** (`' A'`), so whitespace
stripping is required.

### Verdict

Usable as a **teacher**, not as a server. 15.18 GiB of 15.98 leaves 0.80 GiB on a machine that is
also a daily desktop — acceptable for a bounded offline run, not for an always-resident service.
Scoring must go through `/completion`, never `/v1/chat/completions`.

---

## 2026-09-19 — 3.6 vs 3.8 head to head, and a method problem bigger than the result

Both models, **same `/completion` endpoint**, same 1,521 scoreable test rows, `top_n=100`,
temperature fitted on val. The earlier 3.6 number (0.6443) came from `/v1/chat/completions` and was
**not** comparable — the chat template changes the prompt, so endpoint and model were confounded.

### The endpoint mattered more than the model

Same model (3.6), same rows, only the endpoint changed:

| task | via chat | via completion | delta |
|---|---|---|---|
| **ag_news** | 0.8900 | **0.3750** | **−0.515** |
| tickets_priority | 0.0875 | **0.3375** | **+0.250** |
| mmlu | 0.6920 | 0.8120 | +0.120 |
| go_emotions | 0.5881 | 0.6813 | +0.093 |
| tickets_language | 0.9125 | 0.9125 | 0.000 |
| tickets_type | 0.6714 | 0.6571 | −0.014 |
| yelp_score | 0.6250 | 0.6150 | −0.010 |

**Letter-based scoring is not robust to prompt format**, and the sensitivity is task-dependent.
Any cross-model comparison must hold the endpoint fixed; the first one here did not.

### Head to head

| | 3.6-35B-A3B Q4_K_XL | 3.8-27B IQ4_XS |
|---|---|---|
| accuracy, all scored rows | **0.6467** | 0.6299 |
| **accuracy, clean rows only** | **0.6530** | 0.6187 |
| ECE calibrated | 0.1754 (T=5.30) | **0.1056** (T=3.35) |
| rows with a missing option | 87 (5.7%) | 120 (7.9%) |
| p50 latency (4-way) | 2,167 ms | **1,844 ms** |
| throughput | 1.6 /s | **2.1 /s** |

3.6 wins accuracy, 3.8 wins calibration, latency and throughput. 3.8 needs a *lower* temperature
(3.35 vs 5.30), i.e. it is the less overconfident of the two.

### Oracle per-task routing — the multi-teacher question, measured

| | 3.6 | 3.8 | oracle (best per task) | gain |
|---|---|---|---|---|
| all scored rows | 0.6467 | 0.6299 | 0.6980 | **+0.0513** |
| clean rows only | 0.6530 | 0.6187 | 0.6885 | **+0.0356** |

**Treat +0.0356 as an upper bound, and a fragile one.** It is driven almost entirely by `ag_news`
(3.6 0.4082 vs 3.8 0.7405). Strip that one task and routing buys almost nothing. A gain sourced
from a single task, on the task where the method is least stable, is not evidence that per-task
teacher routing generalises.

### The truncation artifact, and why the instrumentation earned its keep

`ag_news` for 3.6 had **26.5% of rows with a missing option** — 4× any other task — and the model
predicted option 0 ("World") on **82.5%** of rows against a near-uniform true distribution
(49/47/50/54). When B/C/D fall outside `top_n`, they take the `-60.0` sentinel and whichever letter
survived wins by default.

AGENTS.md rule 2 says never renormalise over the options you happened to receive. The harness
**recorded** the truncation, which is the only reason this was visible — but the accuracy metric
still consumed those rows. Recording a defect is not the same as excluding it.

3.8 is not immune: it lost 57 of 250 mmlu rows (23%) to truncation.

### What this says about the method

Letters are the problem. After `Answer:`, a raw continuation wants prose — for `ag_news` it wants
to write "World", not "A" — so the letter probabilities sink below any fixed `top_n`. The robust
fix is **sequence scoring**: score the option TEXT appended to the prompt rather than a letter
standing in for it. That removes the truncation window, removes the >26-options limit, and removes
the letter-versus-word conflict in one change. It costs one forward pass per option instead of one
per row, which for an offline teacher is affordable.

---

## 2026-09-19 — Sequence scoring on 3.6: the METHOD matters more than the MODEL

Exact `logP(option | prompt)` by teacher forcing, `top_n=2000`, `--max-options 16`,
single-threaded, T fitted on val. All **1,751** test rows — banking77 and tickets_queue included
for the first time.

### Neither method wins overall; they fail on opposite tasks

On the 1,520 rows both methods could score:

| | accuracy |
|---|---|
| letter | **0.6467** |
| sequence [total logprob] | **0.6449** |
| sequence [mean logprob] | 0.5988 |

Statistically a tie — and that tie hides two large, cancelling effects:

| task | letter | seq[total] | delta |
|---|---|---|---|
| **ag_news** | 0.3750 | **0.6850** | **+0.3100** |
| go_emotions | 0.6813 | 0.7207 | +0.0394 |
| yelp_score | 0.6150 | 0.6300 | +0.0150 |
| tickets_type | 0.6571 | 0.6571 | 0.0000 |
| tickets_priority | 0.3375 | 0.3250 | −0.0125 |
| tickets_language | 0.9125 | 0.8000 | −0.1125 |
| **mmlu** | **0.8120** | 0.4798 | **−0.3322** |

Both mechanisms are real, not defects:

- **MMLU is natively a letter exam.** The model has seen millions of A/B/C/D items; asking it to
  continue "Answer:" with the full text of an option is out of distribution for how it learned the
  task. Letters win by 0.33.
- **ag_news options are natural continuations.** "World", "Sports", "Business" are exactly what
  follows "Answer:" in raw text, while the letters sink below the window. Sequences win by 0.31.

### The result that reframes the multi-teacher question

| oracle | best single | oracle | gain |
|---|---|---|---|
| over **scoring methods** (one model, per task) | 0.6467 | **0.7061** | **+0.0594** |
| over **models** (one method, per task) | 0.6530 | 0.6885 | +0.0356 |

**Choosing how to ask beats choosing whom to ask.** With one model, per-task method selection buys
0.0594; with one method, per-task model selection buys 0.0356 — and that smaller number was
inflated by an `ag_news` truncation artifact, while these two are explicable.

Unlike model routing, the method is plausibly selectable **a priori**: is this a natively
multiple-choice exam (letters) or a set of short natural labels (sequences)? That is a property of
the task, visible without knowing the answer.

### What sequence scoring unlocked

| task | options | accuracy | random floor |
|---|---|---|---|
| banking77 | 77, capped to 16 | **0.5067** | 0.0625 |
| tickets_queue | 52, capped to 16 | **0.4500** | 0.0625 |

Letter scoring could not score these at all. **Caveat that travels with both numbers: the option
set was capped at 16**, so they are easier than the full task.

### Cost and residual error

29,289 token lookups for the test split; **9.32% fell outside the 2,000-window** and took the
window-minimum bound. Row-level that is 27.6%, which is the wrong denominator — a 16-option row
makes ~48 lookups, so at least one miss is near-certain.

`total` needs T=6.657 and `mean` needs T=2.194 — summed logprobs have a far wider spread, so the
two variants are not interchangeable even before accuracy is considered. `total` wins accuracy
(0.6244 vs 0.5546 over all 1,749 rows); `mean` wins calibration (ECE 0.0764 vs 0.1710).

---

## 2026-09-19 — SNI generality benchmark built: held-out predicates for the first time

`git clone --depth 1 https://github.com/allenai/natural-instructions` → `upstream/natural-instructions`
(3.9 GB; gitignored, never vendored), then `harness/build_sni.py --verify` (seed 0) → `data/sni/{train,val,test}.json` + `manifest.json`.

The shipped corpus has 36 question strings, all of them in train AND test — nothing there
distinguishes a general scorer from nine memorised classifiers. This split fixes that: the test
predicates are genuinely unseen.

### Verified category-disjoint split

| | count |
|---|---|
| train tasks | 756 |
| test tasks | 119 |
| categories, train ∩ test | **∅** (60 vs 12 — asserted, not assumed) |
| val | 5 whole train categories (Sentiment Analysis, Toxic Language Detection, Commonsense Classification, Text Categorization, Text Matching), disjoint from test |

### Test: 39 tasks, 7,696 rows, 6 categories

Answerability Classification, Cause Effect Classification, Coreference Resolution, Dialogue Act
Recognition, Textual Entailment, Word Analogy. Label sets {2, 3, 5}; question = Definition[0],
state = instance input, options = sorted(distinct outputs), answer_index = the instance's label.
39 of 55 small-label tasks survived:

| excluded from the 55 | count | why |
|---|---|---|
| input-pointer label sets | 6 | winogrande, COPA ×2, GAP, ROCStories, spolin — labels point at options *in the input*, so a global option set is meaningless |
| CC BY-NC[-SA] licenses | 10 | anli ×3, imppres, daily_dialog ×3, sick ×2, ohsumed |

**Trap, recorded before someone falls in it:** the first license screen used `"NC" in license.upper()`,
which matches the letter sequence inside "licence" and "OANC" — it silently excluded MNLI ×3 and
MultiRC, all valid permissive-license tasks. Fixed to a `CC BY-NC` regex; the false positives are
named in `harness/build_sni.py` and AGENTS.md rule 12.

### Train/val volume

111 train tasks / 39,652 rows, 120 val tasks / 4,800 rows (≥2 labels, semantic sets only, single
answer, inputs p95 < 4k chars; 231/756 train tasks were eligible before the val carve). The
default PyPI-style pitfall does not apply here — no GPU involved.

### What this enables

A LoRA fit on the 111 train tasks, temperature on val, then an accuracy/ECE/Brier number on 39
never-seen predicates — the first number that can distinguish the architecture from memorisation.
**State the contamination caveat in any result:** every SNI task has been public since 2022 and in
the Flan Collection since 2023. Held-out-ness is enforced by *our* partition, not by the model's
ignorance.

---

## 2026-09-19 — SNI fine-tune LAUNCHED (CPU). It teaches the encoder to score option text.

`system_one.py train --base-model mlx-community/gemma-3-270m-bf16 --local-data-dir data/sni_ds
--out-dir results/sni_fit --batch-size 4 --max-options 8 --max-len 128 --epochs 1 --lr 1e-4
--save-every 500 --eval-test` (pid 1507601). Adapter idempotently checkpointed to
`results/sni_fit` every 500 steps; `--resume-adapter DIR` continues from one.

### The config is a memory constraint, not a preference

The first attempt (batch 8, max_len 256) thrashed: **VmPeak 37 GB, VmSwap 10.3 GB, rchar flat
(swap-faulting, ~0.1 steps/s)** on a 30 GiB box where 20 GiB is already used by the desktop.
Diagnosed via `/proc/<pid>/status` (VmSwap/VmPeak), not `ps`.

Batch 4, max_len 128: **VmPeak 8.4 GB, VmSwap 0, ~0.4-0.6 steps/s** — ~2-3 h for one epoch. The
toxicity of the default config was the memory spike, not the model.

### Scale

| | |
|---|---|
| train rows | 16,185 (111 tasks × 150 cap) |
| epochs | 1 |
| tokens/epoch | ~8.2M (max_len 256) -> ~5.5M at max_len 128 |
| val / test rows | 4,800 / 7,696 |
| LoRA | r=16 alpha=32, 3.8M trainable, modules_to_save=['score'] |

`max_options 8`: test never exceeds 5 options, so capping the training option set costs nothing
at eval and cuts the heavy 13-14-option tasks. `max_len 128`: `encode` always keeps the question
+ option tail intact and truncates only leading state.

Result (accuracy/ECE/Brier on the 39 never-seen test predicates) lands here on completion.

### Baselines the SNI test number must beat (computed from test.json, pre-run)

test = 7,696 rows / 39 tasks: **random floor 0.4641**, **majority-class floor 0.5339**.
Most tasks are binary (2 options), and ~14 are answer-skewed enough that predicting the per-task
mode scores 0.53 - that is the "memorised priors" bar, not chance. task880_schema_guided (5
options, mode share 0.21) is the near-uniform control. LoRA accuracy on these rows is only
meaningful against 0.5339.

Follow-up when this run lands: eval the OLD-corpus fit (upstream/pretrained-scorer, 0.6796 on its
own overlapped test) on SNI test - forward-only, no training. If it transfers, the old fit learned
something general; if it falls to ~chance, the old 0.6796 was task memorisation.

---

## 2026-09-19 — BFCL tool-calling eval benchmark + the GPU split, decided

`harness/build_bfcl.py` from `llamastack/bfcl_v3` (BFCL v3 mirror; the official
gorilla-llm repo no longer ships v3 JSONs; subset counts match the handoff's
verified numbers exactly: multiple=200, live_multiple=1053, live_irrelevance=882).
Ground truth here is `[{function_name: args}]`, user query lives in a nested
`turns` conversation, and one live_multiple row has an empty ground truth
(filtered, counted). Emitted: val 120 (seeded from multiple, T fit), test 1,132,
irr 1,122 with the `None of the above` sentinel. BFCL is eval-only, never trained on.

**GPU split, signed off now:** llama-swap was serving qwen3.6-35b +
translategemma-12b + whisper-gpu in 16 GiB when checked - the operator's
transcription/translation stack. The torch arm (270M fine-tune + evals) is
CPU-only by policy: it fits in ~8.4 GB VmPeak at batch 4/max_len 128 and runs
at ~0.33 steps/s; the GPU has no room and taking any breaks STT. GPU work is
only sane in an explicit operator window with llama-swap stopped; the bench
loop never touches it.


---

## 2026-09-19 — SNI held-out-predicate eval: the generality number (final-layer head)

`harness/bench_loop.py` -> system_one.train (out-dir results/sni_fit, batch 4, max_len 128,
max_options 8, 1 epoch, lr 1e-4, save-every 500). LoRA r=16/alpha=32,
3797632 trainable. T = 6.0 fitted on val (val_nll
0.8218976261719577), applied to test - never fitted on the rows scored. Train throughput:
4048 steps / 195.6 min =
0.345 steps/s on CPU.

Baselines for these same test rows: random 0.4641, majority 0.5339 (computed pre-run from
test.json).

### Fine-tuned SNI scorer, head at final layer

| | n | accuracy | ECE | Brier |
|---|---|---|---|---|
| val (T fit) | 4800 | 0.43166666666666664 | 0.03221354545669504 | 0.5435667117797991 |
| **test** | 7696 | **0.4463357588357588** | 0.03607381752914541 | 0.5366255404307516 |

Per task (test, calibrated):

| task | n | acc | ECE | Brier |
|---|---|---|---|---|
| task020_mctaco_span_based_question | 200 | 0.7550 | 0.2447 | 0.4897 |
| task050_multirc_answerability | 200 | 0.4950 | 0.0118 | 0.5005 |
| task1155_bard_analogical_reasoning_trash_or_treasure | 200 | 0.5150 | 0.0028 | 0.4996 |
| task133_winowhy_reason_plausibility_detection | 200 | 0.5250 | 0.0188 | 0.4995 |
| task1344_glue_entailment_classification | 200 | 0.4750 | 0.0259 | 0.5001 |
| task1388_cb_entailment | 200 | 0.4850 | 0.1396 | 0.6568 |
| task1390_wscfixed_coreference | 200 | 0.4200 | 0.1092 | 0.5111 |
| task1439_doqa_cooking_isanswerable | 200 | 0.4700 | 0.0402 | 0.5016 |
| task1442_doqa_movies_isanswerable | 200 | 0.3900 | 0.1202 | 0.5047 |
| task1529_scitail1.1_classification | 200 | 0.5000 | 0.0170 | 0.5013 |
| task1554_scitail_classification | 200 | 0.4950 | 0.0177 | 0.5006 |
| task1624_disfl_qa_question_yesno_classification | 200 | 0.5350 | 0.0251 | 0.4988 |
| task1640_aqa1.0_answerable_unanswerable_question_classification | 200 | 0.4750 | 0.0372 | 0.5017 |
| task190_snli_classification | 200 | 0.6850 | 0.1782 | 0.4951 |
| task199_mnli_classification | 200 | 0.3150 | 0.1944 | 0.5072 |
| task200_mnli_entailment_classification | 200 | 0.3450 | 0.0057 | 0.6666 |
| task201_mnli_neutral_classification | 200 | 0.3650 | 0.0281 | 0.6659 |
| task202_mnli_contradiction_classification | 200 | 0.3150 | 0.0236 | 0.6666 |
| task226_english_language_answer_relevance_classification | 200 | 0.5250 | 0.0181 | 0.4994 |
| task232_iirc_link_number_classification | 200 | 0.3150 | 0.1895 | 0.5033 |
| task233_iirc_link_exists_classification | 200 | 0.2750 | 0.2303 | 0.5048 |
| task242_tweetqa_classification | 200 | 0.5250 | 0.0186 | 0.4995 |
| task290_tellmewhy_question_answerability | 200 | 0.3250 | 0.1760 | 0.5007 |
| task349_squad2.0_answerable_unanswerable_question_classification | 200 | 0.5000 | 0.0189 | 0.5007 |
| task391_causal_relationship | 200 | 0.4450 | 0.0554 | 0.5001 |
| task392_inverse_causal_relationship | 200 | 0.3300 | 0.1712 | 0.5008 |
| task520_aquamuse_answer_given_in_passage | 200 | 0.5300 | 0.0280 | 0.4997 |
| task640_esnli_classification | 100 | 0.3700 | 0.0277 | 0.6656 |
| task641_esnli_classification | 198 | 0.3687 | 0.0301 | 0.6663 |
| task642_esnli_classification | 200 | 0.2550 | 0.2655 | 0.5208 |
| task738_perspectrum_classification | 200 | 0.4650 | 0.0425 | 0.5013 |
| task828_copa_commonsense_cause_effect | 200 | 0.4950 | 0.0153 | 0.5004 |
| task879_schema_guided_dstc8_classification | 200 | 0.5200 | 0.0105 | 0.4995 |
| task880_schema_guided_dstc8_classification | 200 | 0.1750 | 0.0305 | 0.8005 |
| task890_gcwd_classification | 198 | 0.3384 | 0.0020 | 0.6671 |
| task935_defeasible_nli_atomic_classification | 200 | 0.4650 | 0.0381 | 0.5005 |
| task936_defeasible_nli_snli_classification | 200 | 0.4700 | 0.0331 | 0.5004 |
| task937_defeasible_nli_social_classification | 200 | 0.5450 | 0.0433 | 0.4995 |
| task970_sherliic_causal_relationship | 200 | 0.5700 | 0.0595 | 0.4972 |
**Contamination caveat:** every SNI task has been public since 2022 and in the Flan
Collection since 2023; Gemma 3's pretraining corpus is not itemised to exclude SNI-derived text.
Held-out-ness is enforced by *our* partition of the fine-tune, not by base-model ignorance.

### Transfer comparator: the OLD corpus scorer (0.6796 on its own test) on SNI test

| | n | accuracy | ECE | Brier |
|---|---|---|---|---|
| test | 7696 | **0.4699844074844075** | 0.05712919219005727 | 0.5427489352861766 |

Temperature fitted on SNI val. If this scores near random/majority, the old fit was task
memorisation; if it transfers, it learned something general.


---

## 2026-09-20 — Referee scoreboard: the no-training architectures on identical held-out rows

Fast comparison on the small test (SNI 50/task = 1,950; BFCL 400/400). Every cell is
**calibrated accuracy / ECE**, temperature fitted on each benchmark's val, applied to its test.
No architecture was trained for this table beyond what it already was.

| held-out benchmark | released corpus scorer (causal-LM) | our SNI-LM fit (causal-LM) | Laya (ModernBERT) |
|---|---|---|---|
| SNI test (1,950, held-out predicates) | 0.4723076923076923 / 0.05405508962386975 | 0.4523076923076923 / 0.029199077966692032 | — |
| BFCL test (400, tool calls) | 0.8125 / 0.036272448173168124 | 0.4775 / 0.08540399230347484 | — |
| BFCL irr (400, no-tool slice) | 0.6375 / 0.20076229234194992 | 0.985 / 0.25040618638490636 | — |

Baselines (SNI test): random 0.4641, majority 0.5339. A cell near random = the architecture does
not generalise to unseen predicates; well above 0.5339 = it does.

**Laya in-distribution reproduction check** (its own claimed home turf (AG News, published 0.950), our identical corpus
choice rows): calibrated acc ? / ECE ? on
? rows. If this is far below Laya's published ~0.95 AG News / 0.766, its
held-out column is not trustworthy (the checkpoint + our harness disagree with its own numbers).

The build experiments (layer-8 head, jevlike option-attention) append their columns as they
finish training.


---

## 2026-09-19 — Head at layer 8 vs final layer, on the SAME SNI benchmark

`harness/head8_fit.py` (out-dir results/head8_fit), identical data and config to the SNI fit
except the score head reads hidden_states[8] (Gemma3TextForSequenceClassificationHeadAtLayer).
T = 6.0 fitted on val.

| | n | accuracy | ECE | Brier |
|---|---|---|---|---|
| val (T fit) | 4800 | 0.43145833333333333 | 0.05830068147420683 | 0.5513206889976274 |
| **test, head layer 8** | 1950 | **0.46615384615384614** | 0.030861198349422584 | 0.5406405933066173 |
| test, head final layer (ref, above) | 1950 | **0.4463** | - | - |

Layer-probe precedent: frozen-feature accuracy peaked at layer 8 (0.5066) and the final layer was
0.0497 worse on the old corpus test. This answers whether the same holds on held-out predicates.

---

## 2026-09-20 — Reflex-decisions benchmark + the GPU (35B) "agent" baseline

The tasks a System One model is actually for - fast structured judgments, not deliberation.
`harness/build_reflex.py`: MATH topic (choice/7 areas), MATH level (score/1-5), code defect
(noul, "is this C function vulnerable?"). Sources: `nlile/hendrycks-MATH-benchmark` (MIT),
`google/code_x_glue_cc_defect_detection` (Devign). train 6,000 / val 600 / test 600 (200/task),
every option meaningful (topic name / level / yes-no). Held out by DOMAIN.

### GPU baseline - how the big model answers, via llama-swap

`baseline_logprob.py --data-dir data/reflex --model qwen3.6-35b --endpoint completion --workers 1`
(transcription released for the run). Letter-logprob scoring, T=6.0 fitted on val. Test (600):

| task | n | acc | random floor | read |
|---|---|---|---|---|
| math_topic | 200 | **0.485** | 0.143 | real skill - the 35B recognises the area |
| math_level | 200 | **0.220** | 0.200 | at chance - it cannot judge difficulty |
| code_defect | 200 | **0.450** | 0.500 | below chance - logprob scoring can't spot vulns |
| ALL | 600 | 0.385 | - | ECE 0.096, p50 latency **1163 ms** |

So even a 35B, prompted, is only good at *one* of these reflexes (topic recognition) and useless
at the other two. That is the bar a trained 270M scorer has to beat - the next stage trains on
the reflex train split and re-evaluates.

---

## 2026-09-20 — Reflex reeval: small+trained beats big+prompted, and domain transfer fails

270M LoRA fine-tuned on the reflex train split (1 epoch, 1500 steps, 104 min CPU, T=1.1 -
naturally calibrated, no grid saturation). Test (600), calibrated accuracy:

| task | chance | 35B prompted (logprob) | SNI-scorer zero-shot (held-out DOMAIN) | **270M trained (in-domain)** |
|---|---|---|---|---|
| math_topic | 0.143 | 0.485 | **0.095** | **0.695** |
| math_level | 0.200 | 0.220 | 0.105 | **0.345** |
| code_defect | 0.500 | 0.450 | 0.565 | 0.550 |
| ALL | - | 0.385 | 0.255 | **0.530** |

Three findings:
1. **Small + trained wins.** The in-domain 270M (0.530) beats the prompted 35B (0.385) on every
   task - the project's thesis, now on Jev-relevant reflex decisions. It even beats the 35B on
   code_defect where the big model sat *below* chance.
2. **Cross-domain transfer fails.** The SNI-trained scorer (generic NLI/sentiment) zero-shot on
   reflex scores 0.255 - and is *below chance* on math (0.095 / 0.105): its learned priors
   actively mislead on an unseen domain. Consistent with the held-out-predicate collapse. These
   scorers memorise their training distribution; they do not generalise across domains.
3. **Binary transfers, structured labels don't.** code_defect (yes/no) transfers fine zero-shot
   (0.565, ~= the trained model); math (topic/level with domain-specific label sets) does not.

Implication for the build: train on the *target* decision distribution. A single scorer won't
generalise across domains zero-shot at this scale - which is why Jev is trained on diverse
decision data and the narrow clones don't transfer.

---

## 2026-09-20 — Why the CPU cooks under GPU-only QLoRA, and the fix

Symptom: during 7B QLoRA the 7600X sat at 95C and spiked to 104C (past Tjmax);
the guardian tripped. GPU was fine throughout (junction ~92, mem ~97; crit 110/108).

Diagnosis (measured, not theorised):
- Stopping training dropped CPU 104C -> 68C in 15s: the heat is the workload, not
  ambient. Per-thread sampling INSIDE the container during a step showed ~2 python
  threads pinned at ~90% each - host-side kernel dispatch + host<->device sync in
  the training loop keep a couple of cores busy. Zen4 then boosts to its 95C
  ceiling and overshoots.
- Anti-spin env (HSA_ENABLE_INTERRUPT=1, OMP_WAIT_POLICY=passive, OMP_NUM_THREADS=4)
  barely moved the pinned threads - the spin is inside torch/ROCm dispatch, not
  OpenMP. Kept as belt-and-suspenders; not the fix.
- GPU power cap only spans 220->198W (10%): minor.

Fix: training is GPU-bound, so cap the CPU max frequency. amd-pstate
scaling_max_freq 5.46GHz -> 4.2GHz dropped CPU 100C -> 74C for ~5% throughput
(capped 7B step 15.0s vs 14.0s uncapped). train_guardian.py applies this at start
and lowers it another 0.4GHz per repeated thermal trip (floor 3.0GHz) so it
self-converges on a hot day instead of giving up.

Also switched the base from Qwen2.5-7B to Qwen3-4B-Instruct-2507: newer (2025),
smaller (33M/4.05B LoRA), cooler, ~11s/step; better fit for single-letter logprob
scoring at less thermal/VRAM cost.

---

## 2026-09-20 — The built model: Qwen3-4B QLoRA beats/matches much larger models

Trained Qwen3-4B-Instruct-2507 with QLoRA (4-bit NF4, r=16, ~33M trainable) on the mixed
decision corpus (SNI train + reflex train = 22,185 rows, prompt -> ' LETTER'), 1 epoch on the
RX 7900 GRE. Scored with hf_score.py (same letter-logprob path as the baselines), T fit on val.

| benchmark | floor | prior open best | 35B prompted | 270M trained | **4B trained (ours)** |
|---|---|---|---|---|---|
| SNI held-out-predicate (1950) | maj 0.534 | 0.472 | - | - | **0.672** |
| reflex ALL (600) | - | - | 0.385 | 0.530 | **0.580** |
| BFCL tool-calling (399) | - | - | 0.937 | - | **0.937** (zero-shot) |

reflex per task (calibrated test acc):
| task | 35B | 270M | 4B |
|---|---|---|---|
| math_topic | 0.485 | 0.695 | **0.785** |
| math_level | 0.220 | 0.345 | **0.410** |
| code_defect | 0.450 | 0.550 | 0.545 |

Three headline results:
1. **SNI held-out-predicate 0.672** vs the best prior open scorer 0.472 (+0.20 absolute, +42%
   relative) and majority 0.534. The generality benchmark that broke every small scorer is now
   clearly beaten by a 4B - broad pretraining + decision-format SFT generalises where 270M
   encoders did not.
2. **BFCL tool-calling 0.937 == the 35B's 0.9373**, and ZERO-SHOT: BFCL is held out of training
   by policy, yet the 4B matches a ~9x larger prompted model on tool selection.
3. **reflex 0.580 > 35B 0.385 and > 270M 0.530**, winning both math tasks outright.

So the built 4B is the strongest open decision scorer we have measured: it beats the prompted
35B on reflex, matches it zero-shot on tool-calling, and lifts SNI generality far past all prior
open scorers - at a size that fits one 16GB card with a QLoRA adapter.

---

## 2026-09-22 - OADK tool routing: the menu shape decides the answer, and scope is not a class

**Setup.** OADK (an agent app for OutSystems 11) exposes 8 MCP tools to an LLM agent:
`render_screen`, `audit`, `build_screen`, `entities`, `create_entity`, `edit_widget`, `edit_css`,
`save`. The plan is to let the System-One scorer route to them. Two questions were open: can it
pick the right tool, and can it refuse a request the toolset cannot serve at all.

`harness/test_oadk_tools.py` already measured question 1 on one fixed 9-option menu (8 tools +
`None of the above`): adapter **33/44 = 0.750**, base **6/44 = 0.136**, CPU fp32, mean 4767 ms,
p50 4122 ms, p90 5181 ms, max 23485 ms. This entry is about what that single number hides.

Everything below is `harness/probe_oadk_menus.py` (new), same protocol as the other harnesses
(prompt ends `\n\nAnswer:`, each option's ` LETTER` first-token log-prob, argmax = pick, softmax
over those log-probs = confidence). Qwen3-0.6B + `06b`, CPU fp32. Raw results:
`results/probe_oadk_menus.json`.

**Per-tool, on the fixed 9-option menu** (n=44, adapter):

| tool | correct | tool | correct |
|---|---|---|---|
| `entities` | 4/4 | `render_screen` | 3/5 |
| `create_entity` | 5/5 | `edit_widget` | 4/5 |
| `edit_css` | 5/5 | `build_screen` | 4/6 |
| `audit` | 3/4 | `save` | **2/4** |
| | | `None of the above` | 3/6 |

No tool is unroutable. `save` and the abstention option are the two weak points - and both are
"write" decisions, which is the dangerous half.

### 1. The same four requests, five menu shapes

Four styling requests, all of which the full menu handles (3/4). Only the menu changes:

| request | full 9 w/ None | 3 w/ None | 3, None first | 2, `[ew,ec]` | 2, `[ec,ew]` |
|---|---|---|---|---|---|
| Recolour the navbar to #ff4fa0. | ok `edit_css` .88 | `None` .99 | `None` .94 | ok .72 | ok .92 |
| Change the theme colour of the top bar. | ok `edit_css` .75 | `None` .99 | `None` .96 | **`edit_widget` .57** | ok .82 |
| The submit button says 'Submit' - make it say 'Save'. | `None` .53 | `None` 1.00 | `None` .98 | ok .94 | ok .98 |
| Set Visible=false on the Alert widget in Pedidos. | ok .84 | `None` .99 | `None` .95 | ok 1.00 | ok 1.00 |
| **total** | **3/4** | **0/4** | **0/4** | **3/4** | **4/4** |

Two things this isolates:

- **`None of the above` is a semantic attractor, not a neutral marker.** Put it in a 3-option menu
  and it takes every request at 0.94-1.00, including ones the 9-option menu routes correctly.
  It is not position or letter bias: it wins at letter C *and* at letter A.
- **Small menus are order-sensitive.** With the sentinel gone, 2 options score 3/4 or 4/4 depending
  only on which tool is listed first - the same pair, swapped. One request flips
  (`edit_css` .82 -> `edit_widget` .57).

Also note the third row: the 2-option menus get right the one request the full 9-option menu gets
*wrong* (0.53 -> 0.98). Small menus are not strictly worse; they are differently wrong.

### 2. Requests no tool can serve are answered confidently

Six requests the toolset cannot satisfy (publishing, deploying, adding a widget, wiring logic,
deleting a screen, offline/mobile), against three no-sentinel phase menus:

| request (must be refused) | lifecycle | styling | data | max conf |
|---|---|---|---|---|
| Publish the module to the platform. | `save` **.91** | `edit_css` .64 | `create_entity` .81 | .91 |
| Deploy this to production. | `save` **.84** | `edit_widget` .52 | `create_entity` .79 | .84 |
| Add a button to the Pedidos screen. | `build_screen` **.92** | `edit_widget` .80 | `create_entity` .83 | .92 |
| Wire the Save button to create the Order entity. | `build_screen` .82 | `edit_widget` **.94** | `create_entity` .81 | .94 |
| Delete the Login screen. | `render_screen` .77 | `edit_widget` .76 | `entities` .55 | .77 |
| Make the app work offline on mobile. | `render_screen` .65 | `edit_css` .64 | `create_entity` .75 | .75 |

Every one of the 18 decisions lands on a real tool; none abstains. Confidence range .75-.94,
against .72-1.00 for *correct* in-scope picks in the same menus (section 1) - **the distributions
overlap, so no confidence threshold separates "right tool" from "no tool exists".**

Removing the sentinel made this worse, not better: `Publish` scored `save` at .79 with the
sentinel present and **.91** without it; `Deploy` .73 -> **.84**. The sentinel's probability mass
redistributes onto the most plausible tool.

### 3. Asking scope as its own question does not help

Second framing: a separate Y/N question, "Can any of the following tools carry out this request?",
with the tools as context rather than as options:

| expected | result |
|---|---|
| out of scope (n=6) | **5 of 6 answered "Yes"** at 0.63-0.98 |
| in scope (n=6) | 6 of 6 answered "Yes" at 0.74-0.98 |

The one catch is `Delete the Login screen` (P(No) = 0.99) - the only request whose verb has no
near neighbour among the tools. `Publish`/`Deploy` sit next to `save` and are affirmed as doable
at 0.93/0.88. A confident answer to a scope question is not evidence of scope discrimination.

### 4. Per-tool confirmation

Third framing: route first (full 8-tool menu, no sentinel), then ask about the winner *alone* -
"is `<tool>` the right tool for this request?", plus a paraphrase, "does `<tool>` do what this
request asks for?".

| request | expect | candidate (stage 1) | "is X the right tool?" | "does X do what this asks?" |
|---|---|---|---|---|
| Publish the module to the platform. | out | `save` **.99** | **Yes** .75 | **Yes** .88 |
| Deploy this to production. | out | `save` **.96** | **Yes** .63 | **Yes** .72 |
| Add a button to the Pedidos screen. | out | `edit_widget` .51 | **Yes** .93 | **Yes** .94 |
| Wire the Save button to create the Order entity. | out | `create_entity` .65 | **Yes** .89 | **Yes** .95 |
| Delete the Login screen. | out | `render_screen` .56 | No .93 | No .97 |
| Make the app work offline on mobile. | out | `render_screen` .57 | Yes .54 | No .52 |
| Recolour the navbar to #ff4fa0. | in | `edit_css` .96 | Yes .93 | Yes .97 |
| Set Visible=false on the Alert widget in Pedidos. | in | `edit_widget` .99 | Yes .62 | Yes .74 |
| Write the module to disk. | in | `save` .96 | **No .76** | **No .71** |
| Run the security audit over the whole module before we publish. | in | `audit` .82 | Yes .73 | Yes .93 |
| What attributes does the Order entity have? | in | `entities` .80 | Yes .95 | Yes .97 |
| Create a login screen called LoginV2. | in | `build_screen` .83 | Yes .90 | Yes .91 |

| framing | overall correct | out-of-scope rejected |
|---|---|---|
| "is X the right tool?" | 6/12 | **1/6** |
| "does X do what this asks?" | 7/12 | **2/6** |

Both fail, and the second failure mode is the worse one: the confirmation **rejects the correct
routing of an in-scope request** - `Write the module to disk` -> candidate `save` at .96 is told
"No, that is not the right tool" at .76. A gate that blocks legitimate saves is worse than no gate.

Note also that with the sentinel removed, stage 1 is *more* confident on the out-of-scope requests
(`Publish` -> `save` at **.99**, `Deploy` -> `save` at **.96**) than it was with a sentinel in the
menu (.79/.73). Every attempt to make the model say "no" by reshaping the question made it say
"yes" more loudly.

### What the scorer can and cannot do (as measured)

| capability | verdict |
|---|---|
| choose among 2-3 valid options, no sentinel | works - 3-4/4, .72-1.00 |
| choose among 9 options including a sentinel | works, weakly - 33/44 = .750 |
| menu that includes an abstention sentinel, small | **fails** - 0/4, sentinel takes everything |
| refuse a request no tool can serve | **fails** - 0/6, confident wrong answers .75-.94 |
| scope as a separate Y/N question | **fails** - 5/6 affirmed as in scope |
| per-tool confirmation of the winner | **fails** - 1/6 out-of-scope rejected, and it rejects a correct in-scope `save` |

The pattern is consistent: this adapter is a **chooser among given options**, not a **gatekeeper
for whether an option exists**. Scope is not a class it was trained on, so no prompt shape
recovers it - which is the same conclusion as the cross-domain transfer result above, applied to
a distribution the model has never seen.

### Implications for training (what to add)

1. **Menu-shape augmentation.** The corpus should vary option count *and* order, including
   2-option and 3-option menus, and both orders of a plausible pair. Today a model tuned at one
   fixed option count does not transfer to a deployment that varies it.
2. **Abstention examples in small menus, balanced.** The sentinel must not become a prior-dominant
   class. Include small menus where the correct answer *is* a real tool, with the sentinel present,
   or the model learns "small menu -> None".
3. **Scope-boundary rows, with the toolset in the prompt.** The 6 verbatim out-of-scope requests
   above are a start; they need to be balanced against in-scope requests that *look* similar
   (`Publish the module` vs `Save the module`; `Add a button` vs `Build a screen`) or the model
   will learn surface keywords instead of the boundary.
4. **Hard negatives for the write path.** `save` scored 2/4, and `Publish`/`Deploy` both land on
   it. If any single routing error can destroy work, it is this one, so it deserves the most rows.
   Note the shape of the failure: the model accepts `save` for "Publish the module" (.99) and
   *rejects* it for "Write the module to disk" (.76 No). It does not have a usable representation
   of what `save` does - the description is in the prompt, but the decision is not grounded in it.
5. **Teach the description, not just the label.** Rows that ask about one named tool and its own
   description (the section-4 framing) are cheap to generate and directly target this. Right now
   that framing is *worse* than the menu framing, which means the model has learned "which option
   looks most like the request" rather than "what does this tool do".
6. **Eval must report menu shape and per-tool recall**, not one aggregate. The 0.750 above is
   compatible with both "usable" and "0/4 on the styling menu"; only the breakdown distinguishes
   them.

### Caveats

- n is small (4-18 decisions per experiment) and single-seed. Treat these as directional, not as
  benchmarks - the same caveat as every other number in this file.
- One model size, one adapter (`06b`, CPU fp32). The 4B is untested here and may behave
  differently; per the earlier entries it is stronger on tool-calling, so it is worth re-running
  `probe_oadk_menus.py --exp all` against it before concluding anything about the family.
- Latency in this file is CPU fp32 with the option list in the prompt, so it tracks prompt length:
  ~4.2-5.1 s at 9 options vs ~2.0-2.9 s at 2-3. Not comparable to the GPU figures elsewhere.
- "Out of scope" is defined by OADK's documented toolset gaps, not by an exhaustive study of what
  the tools can do.

