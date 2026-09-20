"""FINDINGS.md entry rendering shared by the autonomous bench loop.

Each function appends one condition-attached markdown block so a number never
ships without its conditions (AGENTS.md rule 4).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINDINGS = ROOT / "docs/FINDINGS.md"


def _metrics(m):
    m = m or {}
    tc = m.get("test_calibrated") or {}
    vc = m.get("val_calibrated") or {}
    t = tc.get("ALL", {})
    v = vc.get("ALL", {})
    per = "".join(
        f"| {task} | {tc[task]['n']} | {tc[task]['acc']:.4f} | {tc[task]['ece']:.4f} | {tc[task]['brier']:.4f} |\n"
        for task in sorted(k for k in tc if k != "ALL")
    )
    return v, t, per, m


def append_sni(m):
    v, t, per, m = _metrics(m)
    entry = f"""

---

## 2026-09-19 — SNI held-out-predicate eval: the generality number (final-layer head)

`harness/bench_loop.py` -> system_one.train (out-dir results/sni_fit, batch 4, max_len 128,
max_options 8, 1 epoch, lr 1e-4, save-every 500). LoRA r={m.get('lora_r')}/alpha=32,
{m.get('n_trainable')} trainable. T = {m.get('temperature')} fitted on val (val_nll
{m.get('val_nll')}), applied to test - never fitted on the rows scored. Train throughput:
{m.get('steps')} steps / {round(m.get('train_seconds', 0)/60, 1)} min =
{round(m.get('steps_per_sec', 0), 3)} steps/s on CPU.

Baselines for these same test rows: random 0.4641, majority 0.5339 (computed pre-run from
test.json).

### Fine-tuned SNI scorer, head at final layer

| | n | accuracy | ECE | Brier |
|---|---|---|---|---|
| val (T fit) | {v.get('n', '?')} | {v.get('acc', '?')} | {v.get('ece', '?')} | {v.get('brier', '?')} |
| **test** | {t.get('n', '?')} | **{t.get('acc', '?')}** | {t.get('ece', '?')} | {t.get('brier', '?')} |

Per task (test, calibrated):

| task | n | acc | ECE | Brier |
|---|---|---|---|---|
{per}**Contamination caveat:** every SNI task has been public since 2022 and in the Flan
Collection since 2023; Gemma 3's pretraining corpus is not itemised to exclude SNI-derived text.
Held-out-ness is enforced by *our* partition of the fine-tune, not by base-model ignorance.
"""
    _append(entry)


def append_transfer(cal):
    t = (cal or {}).get("ALL", {})
    entry = f"""
### Transfer comparator: the OLD corpus scorer (0.6796 on its own test) on SNI test

| | n | accuracy | ECE | Brier |
|---|---|---|---|---|
| test | {t.get('n', '?')} | **{t.get('acc', '?')}** | {t.get('ece', '?')} | {t.get('brier', '?')} |

Temperature fitted on SNI val. If this scores near random/majority, the old fit was task
memorisation; if it transfers, it learned something general.
"""
    _append(entry)


def append_head8(m):
    v, t, per, m = _metrics(m)
    # layer-18 headline from the earlier SNI metrics for the comparison
    sni_acc = None
    p = ROOT / "results/sni_fit/metrics.json"
    if p.exists():
        try:
            sni_acc = (json_load(p).get("test_calibrated") or {}).get("ALL", {}).get("acc")
        except Exception:
            pass
    entry = f"""

---

## 2026-09-19 — Head at layer 8 vs final layer, on the SAME SNI benchmark

`harness/head8_fit.py` (out-dir results/head8_fit), identical data and config to the SNI fit
except the score head reads hidden_states[8] (Gemma3TextForSequenceClassificationHeadAtLayer).
T = {m.get('temperature')} fitted on val.

| | n | accuracy | ECE | Brier |
|---|---|---|---|---|
| val (T fit) | {v.get('n', '?')} | {v.get('acc', '?')} | {v.get('ece', '?')} | {v.get('brier', '?')} |
| **test, head layer 8** | {t.get('n', '?')} | **{t.get('acc', '?')}** | {t.get('ece', '?')} | {t.get('brier', '?')} |
| test, head final layer (ref, above) | {t.get('n', '?')} | {('**%.4f**' % sni_acc) if sni_acc is not None else '?'} | - | - |

Layer-probe precedent: frozen-feature accuracy peaked at layer 8 (0.5066) and the final layer was
0.0497 worse on the old corpus test. This answers whether the same holds on held-out predicates.
"""
    _append(entry)


def json_load(p):
    import json
    return json.loads(Path(p).read_text())


def append_bfcl(results):
    """results: {'old_test': ALL,..., 'head8_irr': ALL} - calibrated ALL rows."""
    row = lambda r: f"| {r.get('n','?')} | {r.get('acc','?')} | {r.get('ece','?')} | {r.get('brier','?')} |"
    entry = f"""

---

## 2026-09-19 — Tool-calling transfer: all scorers on BFCL (eval-only)

Data: `harness/build_bfcl.py` from `llamastack/bfcl_v3` (BFCL v3 mirror; counts match the
handoff's verified numbers, one live_multiple row with empty ground truth filtered and counted).
val = seeded 120 from bfcl_v3_multi (T fit), test = 1,132 (80 remaining multi + 1,052 live_multi),
irr = 1,122 (irrelevance + live_irrelevance with the 'None of the above' sentinel). State = user
query (+ tool defs), options = the row's function names [+ sentinel], answer = ground truth.
BFCL is eval-only: never trained on.

| scorer | split | n | accuracy | ECE | Brier |
|---|---|---|---|---|---|
| released corpus fit | test | {row(results.get('old_test', {}))} |
| SNI fit (final-layer head) | test | {row(results.get('sni_test', {}))} |
| SNI fit (head at layer 8) | test | {row(results.get('head8_test', {}))} |
| released corpus fit | irr (no-tool slice) | {row(results.get('old_irr', {}))} |
| SNI fit | irr | {row(results.get('sni_irr', {}))} |
| SNI fit (head at layer 8) | irr | {row(results.get('head8_irr', {}))} |

Floors computed from the emitted rows: test random {_floor('data/bfcl/test.json')} /
irr random {_floor('data/bfcl/irr.json')}. The irr slice is the most honest calibration check:
the correct answer is *not calling any tool*, and a model that learned 'always pick an option'
fails it by construction.
"""
    _append(entry)


def _floor(path):
    import json
    rows = json.loads((ROOT / path).read_text())
    if not rows:
        return "?"
    return f"{sum(1.0 / len(r['options']) for r in rows) / len(rows):.4f}"

def _append(entry):
    with open(FINDINGS, "a") as f:
        f.write(entry)


def append_gate():
    """The single legitimate stop: xlam train data requires an HF click-through."""
    entry = f"""

---

## 2026-09-19 — Autonomous track complete; the one remaining step is gated

All CPU stages the loop can do autonomously are done and recorded (SNI fit,
transfer comparator, head-8 fit, BFCL tool-calling transfer). The only thing
left for the tool-calling TRAIN side is a human action: `Salesforce/xlam-function-calling-60k`
is gated click-through and `HF_TOKEN` is unset on this box. With a token, run
`harness/build_bfcl.py`-style builder for xlam (or the handoff's xlam notes) and
then the same fit+eval chain. Nothing else is blocked.
"""
    _append(entry)


def append_jevlike(evals, cal=None, variants=None):
    """evals: {'base_sni_test': {model:{...}, shuffled_context:{...}}, 'big_...': ...}
    cal: {'base': {'sni': {'T':..,'splits':{...}}, 'bfcl': {...}}, 'big': {...}}
    variants: [('base',[...],pt), ('big',[...],pt)]"""
    variants = variants or [("base", [], "")]
    def row(name, b):
        if not b:
            return "| - | - | - |"
        m = b.get("model", {})
        s = b.get("shuffled_context", {})
        return (f"| {m.get('top1', '?')} | {m.get('top3', '?')} | {m.get('ece', '?')} "
                f"| control {s.get('top1', '?')} |")
    sni_top1 = None
    p = ROOT / "results/sni_fit/metrics.json"
    if p.exists():
        try:
            sni_top1 = (json_load(p).get("test_calibrated") or {}).get("ALL", {}).get("acc")
        except Exception:
            pass
    variant_blocks = ""
    for tag, extra, pt in variants:
        cal_rows = ""
        for label, c in ((cal or {}).get(tag) or {}).items():
            for name, s in c.get("splits", {}).items():
                cal_rows += (f"| {name} | {s.get('n', '?')} | {s.get('cal_acc', '?')} "
                             f"| {s.get('cal_ece', '?')} | {s.get('cal_brier', '?')} "
                             f"| {s.get('ms_per_row', '?')} | {s.get('k_mean', '?')} |\n")
        variant_blocks += f"""
Variant `{tag}` ({' '.join(extra) if extra else 'default tiny config'}):

| rows | top-1 | top-3 | ECE | shuffled-control top-1 |
|---|---|---|---|---|
| SNI test (7,696) | {row(f'{tag}_sni_test', evals.get(f'{tag}_sni_test'))} |
| BFCL test (1,132) | {row(f'{tag}_bfcl_test', evals.get(f'{tag}_bfcl_test'))} |
| BFCL irr (1,122) | {row(f'{tag}_bfcl_irr', evals.get(f'{tag}_bfcl_irr'))} |

| rows | n | cal acc | cal ECE | cal Brier | ms/row | k mean |
|---|---|---|---|---|---|---|
{cal_rows}"""
    entry = f"""

---

## 2026-09-19 — jevlike: the dedicated option-attention architecture, same rows

`harness/bench_loop.py` stage: `jevlike.train` (tiny byte encoder - a representation learned
from scratch, no pretrained prior), on data/jevlike/sni_train.jsonl (16,185 rows, same splits
as the SNI fit). Mapping: context = question + state, options = row options, label =
answer_index. Protocol per benchmark: T fitted on its val, applied to its tests; CPU latency
at batch 64.
{variant_blocks}Reference: SNI fit (LM-continuation scorer, final-layer head) test accuracy
{('%.4f' % sni_top1) if sni_top1 is not None else '?'} on the same 7,696 rows; the one-pass
shape scored each decision in ~0.2-2 ms on CPU against ~400 ms/decision for the LM arm (eval).
The frozen Qwen2.5-0.5B encoder variant measured ~7 s/row on this CPU (16k rows ~30 h) and is
parked, not omitted. On enough data jevlike's own scratch model beat its frozen-encoder one
(29% vs 26% on target-disjoint Wikispeedia), which is why the improvement path here is capacity
and training length on the scratch representation - no pretrained prior - with the same rows.
"""
    _append(entry)


def _append(entry):
    with open(FINDINGS, "a") as f:
        f.write(entry)


def append_laya(res):
    """res: {'sni': {'T':.., 'val':{...}, 'test':{...}, 'irr':{...}}, 'bfcl': {...}}"""
    def row(b):
        if not b:
            return "| - | - | - | - |"
        return (f"| {b.get('n', '?')} | {b.get('acc', '?')} | "
                f"{b.get('cal_acc', '?')} | {b.get('cal_ece', '?')} | "
                f"{b.get('cal_brier', '?')} | {b.get('ms_per_row', '?')} |")
    blocks = ""
    for bench in ("corpus", "sni", "bfcl"):
        e = res.get(bench) or {}
        blocks += f"""
| {bench} val (T fit) | {row(e.get('val'))} |
| {bench} test | {row(e.get('test'))} |
| {bench} irr | {row(e.get('irr'))} |
"""
    entry = f"""

---

## 2026-09-19 — Laya: open 421M ModernBERT System-1 engine on OUR benchmarks

`harness/laya_eval.py` -> `convaiinnovations/laya-typed-decisions` (Apache-2.0, RLCD-trained),
zero extra training on our side. Protocol: temperature fitted on each benchmark's val, applied
to its test/irr; CPU latency per decision at max rows.

| rows | n | raw acc | cal acc | cal ECE | cal Brier | ms/row |
|---|---|---|---|---|---|---|
{blocks}Laya's own claims for context: 0.766 on the typed-decisions benchmark (same corpus this
repo measures; our released-scorer reference on identical shared rows was 0.7035, the SNI
failures measured 0.4463), ECE 0.081 vs Jev 0.246 claimed, 32.8 ms GPU. Its own honest
limitations: zero-shot ~0.35, choice degrades >20 options, temperature must be fitted per
domain. The SNI column above is the first held-out-predicate measurement of this architecture:
near random = pretrained-encoder prior does not transfer; well above majority (0.5339) = it does.
"""
    _append(entry)


def append_referee(board, laya):
    """board: {'old': {'sni_test':{n,acc,ece},...}, 'sni_lm': {...}}
    laya: {'corpus':{...}, 'sni':{'test':{cal_acc,cal_ece}}, 'bfcl':{'test':..,'irr':..}}"""
    def cell(d, acc_key="acc", ece_key="ece"):
        if not d:
            return "—"
        return f"{d.get(acc_key, '?')} / {d.get(ece_key, '?')}"
    L = laya or {}
    def laya_cell(bench, split):
        e = (L.get(bench) or {}).get(split) or {}
        return cell(e, "cal_acc", "cal_ece")
    rows = [
        ("SNI test (1,950, held-out predicates)",
         cell(board.get("old", {}).get("sni_test")),
         cell(board.get("sni_lm", {}).get("sni_test")),
         laya_cell("sni", "test")),
        ("BFCL test (400, tool calls)",
         cell(board.get("old", {}).get("bfcl_test")),
         cell(board.get("sni_lm", {}).get("bfcl_test")),
         laya_cell("bfcl", "test")),
        ("BFCL irr (400, no-tool slice)",
         cell(board.get("old", {}).get("bfcl_irr")),
         cell(board.get("sni_lm", {}).get("bfcl_irr")),
         laya_cell("bfcl", "irr")),
    ]
    body = "".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n" for r in rows)
    corpus = (L.get("corpus") or {}).get("test") or {}
    entry = f"""

---

## 2026-09-20 — Referee scoreboard: the no-training architectures on identical held-out rows

Fast comparison on the small test (SNI 50/task = 1,950; BFCL 400/400). Every cell is
**calibrated accuracy / ECE**, temperature fitted on each benchmark's val, applied to its test.
No architecture was trained for this table beyond what it already was.

| held-out benchmark | released corpus scorer (causal-LM) | our SNI-LM fit (causal-LM) | Laya (ModernBERT) |
|---|---|---|---|
{body}
Baselines (SNI test): random 0.4641, majority 0.5339. A cell near random = the architecture does
not generalise to unseen predicates; well above 0.5339 = it does.

**Laya in-distribution reproduction check** (its own claimed home turf (AG News, published 0.950), our identical corpus
choice rows): calibrated acc {corpus.get('cal_acc', '?')} / ECE {corpus.get('cal_ece', '?')} on
{corpus.get('n', '?')} rows. If this is far below Laya's published ~0.95 AG News / 0.766, its
held-out column is not trustworthy (the checkpoint + our harness disagree with its own numbers).

The build experiments (layer-8 head, jevlike option-attention) append their columns as they
finish training.
"""
    _append(entry)
