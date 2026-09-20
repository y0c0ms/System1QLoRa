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

## Calibration & comparison

**Do not adopt an upstream repo's headline metrics.** Re-derive on the *test* split with your own
protocol. Published numbers are frequently val-split, on a handful of questions, with temperature
fit in-sample — a best case reported as the result. Fit temperature on validation, apply to test,
never fit on the rows you are scoring.

**Measure the split you claim.** A category-disjoint (held-out-predicate) split and an in-distribution
split answer different questions. Verify `train ∩ test = ∅` on the *predicates*, not just the inputs,
before calling a number "generality."

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
