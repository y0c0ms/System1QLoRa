#!/usr/bin/env python3
"""Layer-wise linear probing of a FROZEN base model.

THE QUESTION. The LoRA scorer reaches 0.7035 on the 1521 comparable test rows.
How much of that was already sitting in the frozen representation, and WHERE?

Method: freeze `gemma-3-270m`, run each (state, question, option) sequence once,
keep the hidden state at the last non-pad token from EVERY layer, then fit one
linear scorer per layer. No LoRA, no gradients through the base. The per-layer
accuracy curve says where decision-relevant information lives; the gap between
the best layer and 0.7035 says what LoRA actually added.

WHY THIS BEFORE A BIGGER CORPUS. If a linear read of frozen features already
recovers most of the accuracy, then the fine-tune is doing little and the
interesting variable is the base model, not the training. That changes what a
larger benchmark should even measure - so it is worth knowing first, and it needs
no new data.

COMPARABILITY. `encode` and `pad_batch` are upstream's, imported rather than
reimplemented, so the probe sees byte-identical inputs to the LoRA scorer. The
protocol is upstream's too: fit on TRAIN, temperature on VAL, report TEST.

FEATURES ARE STANDARDISED per dimension using TRAIN statistics. Layers differ in
activation scale by orders of magnitude; without this the comparison across
layers measures norm, not information.

NO BIAS TERM, deliberately: scores are softmaxed over a question's options, so a
constant added to every option cancels. Upstream's head is bias-free too.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "upstream"))
from system_one import encode, pad_batch  # noqa: E402  upstream, not reimplemented

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
ADAPTER = os.path.join(os.path.dirname(__file__), "..", "upstream", "pretrained-scorer")


def take_per_task(rows, n):
    seen, out = {}, []
    for r in rows:
        t = r["task"]
        if seen.get(t, 0) >= n:
            continue
        seen[t] = seen.get(t, 0) + 1
        out.append(r)
    return out


def build_instances(rows, max_options):
    """Flatten rows into scoring instances, recording which belong together.

    Returns (instances, groups) where each group is (indices, label). Option
    subsetting mirrors upstream's Batcher: keep the gold plus sampled negatives.
    """
    import random
    rng = random.Random(0)
    inst, groups = [], []
    for r in rows:
        opts, ans = list(r["options"]), int(r["answer_index"])
        if max_options and len(opts) > max_options:
            negs = [i for i in range(len(opts)) if i != ans]
            rng.shuffle(negs)
            chosen = [ans] + negs[: max_options - 1]
            rng.shuffle(chosen)
        else:
            chosen = list(range(len(opts)))
        idx = []
        for oi in chosen:
            idx.append(len(inst))
            inst.append((r["state"], r["question"], opts[oi]))
        groups.append((idx, chosen.index(ans), r["task"]))
    return inst, groups


@torch.no_grad()
def extract(model, tok, inst, max_len, batch, tag, n_layers):
    """Per-layer last-non-pad hidden states. Cached - this is the expensive part."""
    path = os.path.join(RES, "feats_%s.npy" % tag)
    if os.path.exists(path):
        f = np.load(path)
        if f.shape[0] == len(inst):
            print("  %s: cached features %s" % (tag, f.shape), flush=True)
            return f
        print("  %s: cache has %d rows, need %d - re-extracting"
              % (tag, f.shape[0], len(inst)), flush=True)

    out = np.zeros((len(inst), n_layers, model.config.hidden_size), dtype=np.float16)
    t0 = time.time()
    for s in range(0, len(inst), batch):
        chunk = inst[s:s + batch]
        seqs = [encode(tok, st, q, o, max_len) for st, q, o in chunk]
        ids, attn = pad_batch(seqs, tok.pad_token_id, max_len)
        hs = model(input_ids=ids, attention_mask=attn,
                   output_hidden_states=True).hidden_states
        last = attn.sum(1) - 1                      # index of the last real token
        rows_ix = torch.arange(ids.shape[0])
        for li, h in enumerate(hs):
            out[s:s + len(chunk), li] = h[rows_ix, last].float().numpy().astype(np.float16)
        if (s // batch) % 20 == 0:
            done = s + len(chunk)
            rate = done / max(1e-9, time.time() - t0)
            print("    %s %d/%d  (%.1f seq/s, eta %.0f min)"
                  % (tag, done, len(inst), rate, (len(inst) - done) / rate / 60), flush=True)
    np.save(path, out)
    print("  %s: extracted %s in %.0fs" % (tag, out.shape, time.time() - t0), flush=True)
    return out


def pad_groups(groups):
    k = max(len(g[0]) for g in groups)
    idx = np.full((len(groups), k), -1, dtype=np.int64)
    lab = np.zeros(len(groups), dtype=np.int64)
    for i, (ix, a, _) in enumerate(groups):
        idx[i, :len(ix)] = ix
        lab[i] = a
    return torch.from_numpy(idx), torch.from_numpy(lab)


def group_logits(scores, idx):
    """Gather flat per-instance scores into (n_questions, max_k), masking padding."""
    g = scores[idx.clamp(min=0)]
    return g.masked_fill(idx < 0, float("-inf"))


def fit_probe(Xtr, gtr, steps=400, lr=0.05):
    idx, lab = pad_groups(gtr)
    X = torch.from_numpy(Xtr).float()
    w = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(group_logits(X @ w, idx), lab)
        loss.backward()
        opt.step()
    return w.detach(), float(loss.detach())


def probs_for(X, w, groups, T=1.0):
    idx, lab = pad_groups(groups)
    lg = group_logits(torch.from_numpy(X).float() @ w, idx) / T
    p = torch.softmax(lg, dim=1)
    return p.numpy(), lab.numpy(), idx.numpy()


def ece(p, lab, idx, n_bins=10):
    valid = idx >= 0
    conf = np.where(valid, p, -1).max(1)
    pred = np.where(valid, p, -1).argmax(1)
    corr = (pred == lab).astype(float)
    out, n = 0.0, len(lab)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        out += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(out)


# Upstream's grid starts at 0.25. Eight layers pinned to exactly that floor -
# they wanted to sharpen FURTHER, and their reported ECE (~0.40) was an
# artifact of the truncated search, not a property of the layer. The floor is
# geometric now so it can reach genuinely sharp temperatures.
T_GRID = [0.02 * (1.0353 ** i) for i in range(200)]   # 0.02 -> ~20
T_LO, T_HI = T_GRID[0], T_GRID[-1]


def fit_T(X, w, groups):
    """Grid-search the temperature, and SAY SO if the optimum is at an edge.

    Upstream's grid stops at 6.0. The smoke test drove an overfit probe to
    exactly 6.00 on nine layers out of nineteen - the search had saturated and
    the reported temperature was the edge of the grid, not the optimum. A
    calibration number taken from a saturated search is meaningless, and nothing
    in upstream's code reports that it happened. The grid is widened here and
    saturation is returned alongside the value.
    """
    best, bnll = T_LO, float("inf")
    idx, lab = pad_groups(groups)
    lg = group_logits(torch.from_numpy(X).float() @ w, idx)
    for t in T_GRID:
        nll = float(torch.nn.functional.cross_entropy(lg / t, lab))
        if nll < bnll:
            bnll, best = nll, t
    saturated = best <= T_LO * 1.001 or best >= T_HI * 0.999
    return best, saturated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mlx-community/gemma-3-270m-bf16")
    ap.add_argument("--max-len", type=int, default=256)   # what the shipped model used
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-per-task", type=int, default=300)
    ap.add_argument("--val-per-task", type=int, default=48)
    ap.add_argument("--max-options", type=int, default=8)  # TRAIN only; eval uses full sets
    # Smoke tests MUST be able to shrink the test set too. The first smoke run
    # shrank train and val but still extracted all 20,632 test sequences - 613
    # seconds to validate a pipeline.
    ap.add_argument("--test-per-task", type=int, default=0, help="0 = full test split")
    a = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    print("loading frozen base %s" % a.base, flush=True)
    model = AutoModel.from_pretrained(a.base, dtype=torch.float32).eval()
    n_layers = model.config.num_hidden_layers + 1
    print("  hidden=%d layers=%d (0 = embeddings, before any block)"
          % (model.config.hidden_size, n_layers), flush=True)

    tr = take_per_task(json.load(open(os.path.join(DATA, "train.json"))), a.train_per_task)
    va = take_per_task(json.load(open(os.path.join(DATA, "val.json"))), a.val_per_task)
    te = json.load(open(os.path.join(DATA, "test.json")))
    if a.test_per_task:
        te = take_per_task(te, a.test_per_task)

    itr, gtr = build_instances(tr, a.max_options)
    iva, gva = build_instances(va, 0)
    ite, gte = build_instances(te, 0)
    print("  train %d rows -> %d seqs | val %d -> %d | test %d -> %d"
          % (len(tr), len(itr), len(va), len(iva), len(te), len(ite)), flush=True)

    Xtr = extract(model, tok, itr, a.max_len, a.batch, "train", n_layers)
    Xva = extract(model, tok, iva, a.max_len, a.batch, "val", n_layers)
    Xte = extract(model, tok, ite, a.max_len, a.batch, "test", n_layers)

    print("\n%-6s %-9s %-9s %-9s %-9s %s"
          % ("layer", "test_acc", "test_ece", "T", "train_loss", "flags"))
    results = []
    for L in range(n_layers):
        F16MAX = 65504.0
        clip = lambda A: np.nan_to_num(A.astype(np.float32),
                                       nan=0.0, posinf=F16MAX, neginf=-F16MAX)
        tr_f = clip(Xtr[:, L])
        mu, sd = tr_f.mean(0), tr_f.std(0) + 1e-6   # TRAIN stats only
        w, loss = fit_probe((tr_f - mu) / sd, gtr)
        va_f = (clip(Xva[:, L]) - mu) / sd
        te_f = (clip(Xte[:, L]) - mu) / sd
        T, sat = fit_T(va_f, w, gva)
        p, lab, idx = probs_for(te_f, w, gte, T)
        acc = float((np.where(idx >= 0, p, -1).argmax(1) == lab).mean())
        e = ece(p, lab, idx)
        print("%-6d %-9.4f %-9.4f %-9.2f %-9.4f %s"
              % (L, acc, e, T, loss, "T-SATURATED" if sat else ""), flush=True)
        results.append({"layer": L, "test_acc": acc, "test_ece": e, "T": T,
                        "T_saturated": sat, "train_loss": loss})

    best = max(results, key=lambda r: r["test_acc"])
    print("\n  best layer: %d  acc=%.4f  ece=%.4f" % (best["layer"], best["test_acc"], best["test_ece"]))
    print("  LoRA scorer on the SAME full test split: acc=0.6796 ece=0.0220")
    print("  gap: %+.4f accuracy" % (best["test_acc"] - 0.6796))
    with open(os.path.join(RES, "layer_probe.json"), "w") as f:
        json.dump({"base": a.base, "layers": results, "best": best}, f, indent=1)


if __name__ == "__main__":
    main()
