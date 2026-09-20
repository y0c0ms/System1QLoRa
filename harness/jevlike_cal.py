#!/usr/bin/env python3
"""Calibrate and measure a jevlike checkpoint with OUR protocol.

The one-pass option scorer is the shape that approaches Jev; this harness makes
its numbers comparable with the LM-continuation arms: temperature fitted on the
benchmark's val, applied to test (never fitted on scored rows), accuracy/ECE/
Brier, plus jevlike's own shuffled-context control and a per-decision latency
measurement (the project's core cost question, with conditions).

Protocol per split-set: T from <name>_val.jsonl -> <name>_test.jsonl and
<name>_irr.jsonl (the irrelevance slice carries the sentinel option).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))
sysp = __import__("sys").path
JEV_DIR = ROOT / "upstream/jevlike"
if str(JEV_DIR) not in sysp:
    sysp.insert(1, str(JEV_DIR))
from upstream.system_one import fit_temperature  # noqa: E402


def softmax(xs):
    m = np.max(xs)
    es = np.exp(xs - m)
    return es / es.sum()


def ece(probs, answers, n_bins=10):
    probs = np.asarray(probs)
    conf = probs.max(-1)
    pred = probs.argmax(-1)
    correct = pred == np.asarray(answers)
    out = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        sel = (conf > lo) & (conf <= hi)
        if sel.sum() == 0:
            continue
        out += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(out)


def brier(probs, answers):
    probs = np.asarray(probs)
    y = np.zeros(probs.shape)
    y[np.arange(len(y)), answers] = 1.0
    return float(((probs - y) ** 2).sum(1).mean())


@torch.no_grad()
def collect(model, loader, device, shuffle=False):
    from jevlike.train import move
    logits, labels = [], []
    t0 = time.time()
    for host in loader:
        b = move(host, device)
        out = model(b, shuffle_context=shuffle).cpu().numpy()
        logits.append(out)
        labels.append(host["labels"].numpy())
    dt = time.time() - t0
    return np.concatenate(logits), np.concatenate(labels), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--val-jsonl", action="append", default=[])
    ap.add_argument("--test-jsonl", action="append", default=[])
    ap.add_argument("--labels", action="append", default=[])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from jevlike.data import JsonlDataset
    from jevlike.model import load_checkpoint, select_device

    device = select_device(args.device)
    model, collator, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    results = {}
    # temperature per split-set from its val
    for label in args.labels:
        val_path = next(p for p in args.val_jsonl if label in p)
        val_logits, val_labels, _ = collect(
            model, DataLoader(JsonlDataset(val_path), batch_size=args.batch_size,
                              collate_fn=collator), device)
        T, nll = fit_temperature([list(map(float, r)) for r in val_logits],
                                 [int(x) for x in val_labels])
        entry = {"T": float(T), "val_nll": float(nll), "splits": {}}
        # tests for THIS label
        for test_path in args.test_jsonl:
            if label not in Path(test_path).name:
                continue
            name = Path(test_path).stem
            lg, lb, dt = collect(model, DataLoader(JsonlDataset(test_path),
                                                   batch_size=args.batch_size,
                                                   collate_fn=collator), device)
            lg_c, _, _ = collect(model, DataLoader(JsonlDataset(test_path),
                                                   batch_size=args.batch_size,
                                                   collate_fn=collator),
                                 device, shuffle=True)
            raw = np.array([softmax(r) for r in lg])
            cal = np.array([softmax(r / T) for r in lg])
            ctl = np.array([softmax(r) for r in lg_c])
            entry["splits"][name] = {
                "n": len(lb),
                "raw_acc": float((raw.argmax(-1) == lb).mean()),
                "raw_ece": ece(raw, lb), "raw_brier": brier(raw, lb),
                "cal_acc": float((cal.argmax(-1) == lb).mean()),
                "cal_ece": ece(cal, lb), "cal_brier": brier(cal, lb),
                "control_top1": float((ctl.argmax(-1) == lb).mean()),
                "ms_per_row": round(1000 * dt / len(lb), 1),
                "k_mean": round(float(raw.shape[1]), 1),
            }
        results[label] = entry

    out = ROOT / "results/jevlike_cal.json"
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1), flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()