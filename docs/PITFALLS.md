# Pitfalls

Each item below cost a wrong number before it became a rule. They are written generally so you can
apply them to your own option-scoring or calibration work, not just this repo. The lab journal in
`FINDINGS.md` has the specific incidents.

## Measurement

**A probe that prints nothing has not succeeded.** Assert on the *thing you actually need* — the
count of returned items — never on the presence of its container. A check that a response key
*exists* passed while the key held an empty list, and reported the route "usable" with zero tokens.

**Never renormalise over only the options you happened to receive.** Top-logprobs lists truncate;
at a small `n` an option is simply absent. Renormalising over the options that came back produces a
clean, confident, *wrong* distribution with nothing to signal the loss. Any missing option is
**reported as missing**; a run with missing options is not a measurement.

**A sampling constraint is not a reporting constraint.** A GBNF/grammar (`root ::= [ABC]`) forces
the *sampled* token but does not change the returned distribution — llama.cpp reports the
**pre-grammar** log-probs, byte-identical to the unconstrained call. Do not reach for a grammar to
fix a truncated-reporting problem.

**Every latency figure ships with its conditions.** Hardware, option count, context length,
repeats, warmup, and whether the model was already resident. A single "~Xms" with none of those
attached is the number this kind of project exists to correct, not to reproduce.

**Score through the raw completion path, not a chat template.** A reasoning/chat template injects
tokens before the answer; letter-scoring reads position 0 and sees `We`, not `A`. Both the baseline
and the trained model must be scored through the *same* raw path or you confound model with prompt.

**An option menu is part of the prompt, not a neutral container.** Adding or removing an option moves
probability mass between the survivors — including onto an abstention sentinel — so the *same*
request can be scored right or wrong purely by menu shape. Measured on the OADK tool menu: four
styling requests that the 9-option menu routes 3/4 collapse to **0/4** when the same tools are
offered as a 3-option menu that includes `None of the above` (every request picked `None` at
0.94–1.00). Reordering the same three options moved the `None` pick from letter C to letter A
without changing the outcome, so it is not a position or letter artifact. **Report the menu
alongside every routing number.**

**Gate batch invariance on decisions, not raw bf16 logits — and prove the gate has teeth.** bf16
GEMM tiling depends on batch shape, so the same row scored alone and in a batch differs by up to
one bf16 ULP at the logit's magnitude (0.5 at |logit| ≥ 64) with zero effect on any decision. A
`max|Δlogit| < 0.5` gate aborted a 9-hour run on exactly 0.5000. Compare option probabilities and
decisive argmax flips instead, then inject the bug the gate exists for (gather at the padded end
instead of each row's last real token): it must fail loudly (here Δp 0.658 and 2/16 flips vs ≤ 0.022
and 0 for the real code).

## Calibration & comparison

**Do not adopt an upstream repo's headline metrics.** Re-derive on the *test* split with your own
protocol. Published numbers are frequently val-split, on a handful of questions, with temperature
fit in-sample — a best case reported as the result. Fit temperature on validation, apply to test,
never fit on the rows you are scoring.

**Measure the split you claim.** A category-disjoint (held-out-predicate) split and an in-distribution
split answer different questions. Verify `train ∩ test = ∅` on the *predicates*, not just the inputs,
before calling a number "generality."

**A confident answer to a scope question is not scope discrimination.** Asking "can any of these
tools carry out this request?" as its own Y/N question does not turn a router into a capability
check. Of six OADK requests no tool could satisfy, five were affirmed as doable at 0.93–0.98;
the only one caught (`Delete the Login screen`, P(No)=0.99) is the one whose verb has no near
neighbour among the tools. `Publish`/`Deploy` sit next to `save` and were routed *to* it at
0.91/0.84 — **higher** than the 0.79/0.73 they scored when an abstention option was present, so
removing the sentinel raised confidence on exactly the cases that must be refused. Scope is not a
class unless it was trained as one.

## Training

**Make the SFT objective identical to the eval metric.** If evaluation reads the log-prob of an
option letter after a specific prompt, train on exactly that prompt → that letter, with loss on the
completion only. Any drift between the training and scoring prompt silently costs accuracy.

**A single-domain scorer does not transfer.** Measured here: a scorer trained on one decision domain
scored *below chance* on another. Train on a mixed corpus spanning every distribution you want to
serve; do not expect zero-shot generalisation across domains at small scale.

**Preserve the decision cue when truncating.** The answer-bearing tail (options + `Answer:`) must
survive truncation, so left-truncate long states rather than cutting the end. Match train and eval
truncation side.

**Train on the menu shapes you will deploy, not just the tool list.** Routing accuracy is a function
of the option set's size, composition *and* order. Measured on the OADK tool menu (four styling
requests, adapter `06b`, CPU): the full 9-option menu scores 3/4; the same tools as a 3-option menu
*including* an abstention sentinel score **0/4** (sentinel first or last, no difference); the same
tools as a 2-option menu without a sentinel score 3/4 or **4/4** depending only on which tool is
listed first. A corpus built at one fixed option count and order does not transfer to a deployment
that varies either — and an abstention sentinel behaves like a class the model has over-learned,
not like a neutral "none" marker.

## Long runs on shared / constrained hardware

**Diagnose "stuck" with `/proc/<pid>/io`, not `ps`.** Under CPU/IO contention a job can look hung
while it is merely starved — sockets open, `rchar` flat. `ps` shows a live process; the IO counters
show whether it is doing anything.

**Any run against a shared or long-lived resource must checkpoint.** Append results per row/step and
flush; key by index so a resumed run skips exactly what is done. Interruption is certain; losing the
work is not. Verify recovery for real (kill mid-run, resume, confirm it continues from the last
checkpoint) rather than trusting that `--resume` works.

**Concurrency is not free — measure it.** More workers made throughput *worse* here (experts on CPU
contended); the right worker count was 1. Measure the specific configuration rather than assuming
parallelism helps.

**A cool, idle-looking GPU can be silently underclocked.** A DPM/power state stuck at "low" ran at a
fraction of speed while *cool* (low temp, low power, slow steps) — the opposite of thermal
throttling. If throughput drops without heat, check the performance level / clocks, not the
temperature.

**Time the real function before theorising about the cause.** Several plausible explanations for a
slowdown were wrong; timing the actual scoring call on one real row gave the answer in one shot.

**Check which attention backend your GPU actually gets before blaming the model size.** On this box
(torch 2.13+rocm7.14, RDNA3/gfx1100) `flash_sdp_enabled()` and `mem_efficient_sdp_enabled()` both
report `True`, yet forcing either one raises *"No available kernel. Aborting execution"* — only
`SDPBackend.MATH` executes. Attention is then materialised in full, so activation memory is O(S²)
and throughput collapses: a 149M encoder needed >5 GB at **8** rows × 512 tokens and managed 39
pairs/s, while a 67M one ran ~7× faster on the same data. This is also why an earlier job peaked at
15 GB and starved the compositor — it was not the batch size. Probe the backend with
`sdpa_kernel(...)` on a tiny tensor before sizing any run, and treat gradient checkpointing as
mandatory rather than an optimisation.

**Cancelling a wrapper job does not necessarily stop the work.** Killing a bash job that runs
`sudo podman exec <ctr> python train.py` kills the wrapper; the container's python can keep running
unnoticed. A stale benchmark here kept burning CPU — and would have shared the GPU — next to the
real training run for half an hour. After any cancel, verify inside the container
(`podman exec <ctr> pgrep -fa <script>`) instead of assuming the work stopped.

**"Non-finite loss" and "finite loss, NaN gradient" are different failures — check which one you
have.** A training run on this stack died with `non-finite loss at step 43`, which reads like a bad
row or a diverging LR. Instrumenting the step instead of guessing showed the loss was *finite* and
only the **gradient** was NaN, from step 16 — a backward-pass problem, not a data one (a full audit
of the corpus found no empty, duplicate or pathological options). The likely trigger is gradient
checkpointing recomputation combined with the MATH-only attention backend on long sequences: the
15-step probe that was stable used short rows and small batches. Check the gradient norm, not just
the loss, before concluding anything about the data.

**A GPU with no preemption needs a duty cycle to be shared.** With only the MATH sdpa backend,
back-to-back kernels monopolise the compute queues and the operator's video stutters even though
the job holds little memory. `--step-sleep` idles the device between steps so other clients get
regular windows; it costs throughput but it is the difference between a usable and an unusable
desktop.
