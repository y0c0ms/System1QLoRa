#!/usr/bin/env python3
"""Tiny-encoder decision specialist.

Cross-encoder formulation, the standard one for encoder-only multiple choice: one
forward pass per (context, option) pair producing a single relevance logit, then
argmax over a row's options. The objective is listwise - a softmax over the row's
option scores with CE against the answer index - so training optimises exactly the
decision the evaluation makes, unlike the decoder trainer which scores one answer
letter per row.

Context handling matches train_decider: the context is truncated from the LEFT (keep
the tail, where the question is) and the option is never truncated.

CPU by default. The operator's GPU belongs to llama-swap (it serves their speech-to-text),
and 22M-150M parameters are tractable without it.

Usage:
  python harness/train_encoder.py --benchmark 20 --base microsoft/deberta-v3-xsmall
  python harness/train_encoder.py --corpus data/distill/train_structured.jsonl \
      --out results/enc_22m --base microsoft/deberta-v3-xsmall --epochs 2
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval_decider import text_field  # canonical field rendering

ROOT = Path(__file__).resolve().parents[1]
OPT_MAX = 64          # options are short; this only bounds pathological rows


def build_context(row):
    """Same field rendering as the scorer - see eval_decider.text_field."""
    return f"State:\n{text_field(row['state'])}\n\nQuestion:\n{text_field(row['question'])}"


def load_rows(corpus, limit):
    rows = [json.loads(l) for l in open(corpus)]
    rows = [r for r in rows if 2 <= len(r["options"]) <= 26
            and 0 <= r["answer_index"] < len(r["options"])]
    if limit:
        rows = rows[:limit]
    return rows


def encode_pair(tok, ctx_ids, opt_ids, max_len):
    """[CLS] ctx [SEP] opt [SEP] with the CONTEXT truncated from the left.

    Built by hand: transformers 5's tokenizer backends no longer expose
    prepare_for_model. verify_pair_layout() checks this against the tokenizer's own
    text_pair encoding at startup, so a model with a different special-token layout
    cannot silently corrupt the training data.
    """
    cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    opt = opt_ids[:OPT_MAX]
    budget = max_len - len(opt) - 3
    ctx = ctx_ids[-budget:] if len(ctx_ids) > budget else ctx_ids
    return [cls_id] + ctx + [sep_id] + opt + [sep_id]


def verify_pair_layout(tok, ctx, opt):
    mine = encode_pair(tok, tok(ctx, add_special_tokens=False)["input_ids"],
                       tok(opt, add_special_tokens=False)["input_ids"], 512)
    theirs = tok(ctx, opt, add_special_tokens=True)["input_ids"]
    if mine != theirs:
        raise SystemExit(f"pair layout mismatch for {tok.__class__.__name__}: "
                         f"built {mine[:12]}... vs tokenizer {theirs[:12]}...")
    return True


def batches_by_budget(chunk, data, plen, tok_budget, max_pairs):
    """Group consecutive rows of a length-sorted chunk so a step's padded tokens stay bounded.

    Padded cost per step is (pairs in step) x (longest pair in step), so a fixed row count
    both wastes compute on short buckets and risks OOM on long ones. Rows are taken until
    either the token budget or the pair cap would be exceeded - the same shape of budget
    eval_decider uses for scoring.
    """
    out, cur, pairs, longest = [], [], 0, 0
    for i in chunk:
        est = max(1, plen[i] // 4)                       # chars -> tokens proxy
        n = data[i]["k"]
        if cur and (pairs + n) * max(longest, est) > tok_budget:
            out.append(cur)
            cur, pairs, longest = [], 0, 0
        cur.append(i)
        pairs += n
        longest = max(longest, est)
        if pairs >= max_pairs:
            out.append(cur)
            cur, pairs, longest = [], 0, 0
    if cur:
        out.append(cur)
    return out


def collate(batch, pad_id):
    """Dynamic padding: pad to the longest sequence in this batch, not to max_len."""
    n = max(len(x) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), n), dtype=torch.long)
    for i, x in enumerate(batch):
        L = len(x)
        ids[i, :L] = torch.tensor(x)
        am[i, :L] = 1
    return ids, am


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "distill" / "train_structured.jsonl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--base", default="microsoft/deberta-v3-xsmall")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-rows", type=int, default=16)
    ap.add_argument("--tok-budget", type=int, default=20000,
                    help="padded tokens per step; bounds both compute and VRAM")
    ap.add_argument("--max-pairs", type=int, default=96, help="cap on option-pairs per step")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--vram-cap-gb", type=float, default=4.0,
                    help="hard per-process cap; the rest of the card stays with the desktop")
    ap.add_argument("--grad-checkpoint", default="auto", choices=["auto", "on", "off"],
                    help="auto = on for cuda. Required here: this ROCm build has only the "
                         "MATH sdpa backend (flash and mem-efficient report no kernel), so "
                         "attention memory is O(S^2) with the full matrix materialised")
    ap.add_argument("--benchmark", type=int, default=0, help="run N steps and report pairs/s, then exit")
    ap.add_argument("--step-sleep", type=float, default=0.0,
                    help="seconds to idle after each step. This GPU has no preemption, so a "
                         "back-to-back training loop monopolises the compute queues and the "
                         "operator's video stutters; sleeping gives other clients regular windows")
    ap.add_argument("--save-steps", type=int, default=200,
                    help="checkpoint interval; rule 9: any run against the shared GPU must be resumable")
    ap.add_argument("--deadline", type=float, default=0.0,
                    help="unix time; stop cleanly once reached and keep the last checkpoint")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if dev == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb / total), 0)
        print(f"device=cuda vram_cap={args.vram_cap_gb}GB of {total:.1f}GB", flush=True)
    else:
        print("device=cpu", flush=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    verify_pair_layout(tok, "State:\npolicy text\n\nQuestion:\nwhat happens?",
                       "processed normally")
    print(f"pair layout verified against {tok.__class__.__name__}", flush=True)

    rows = load_rows(args.corpus, args.limit_rows)
    print(f"base={args.base} rows={len(rows)} pairs={sum(len(r['options']) for r in rows)} "
          f"max_len={args.max_len} threads={args.threads}", flush=True)

    t0 = time.time()
    # contexts are tokenised in full and truncated by hand (left, keeping the tail), so
    # lift the tokenizer's own length check to stop it warning on every long row
    tok.model_max_length = 1 << 30
    data = []
    for r in rows:
        ctx_ids = tok(build_context(r), add_special_tokens=False)["input_ids"]
        pairs = [encode_pair(tok, ctx_ids, tok(o, add_special_tokens=False)["input_ids"], args.max_len)
                 for o in r["options"]]
        data.append({"pairs": pairs, "k": len(pairs), "answer": r["answer_index"],
                     "task": r.get("task"), "source": r.get("source")})
    print(f"tokenised in {time.time() - t0:.0f}s", flush=True)
    # Length bucketing + padded-token budget: char length is a cheap monotone proxy for
    # token length. With only the MATH attention backend, padding cost is quadratic, so
    # random batches that pad to whatever long row they contain are expensive.
    plen = [len(r["state"]) + len(r["question"]) + sum(len(o) for o in r["options"]) for r in rows]
    window = args.batch_rows * 32

    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(dev)
    gc = args.grad_checkpoint == "on" or (args.grad_checkpoint == "auto" and dev == "cuda")
    if gc:
        model.gradient_checkpointing_enable()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params={n_par:,} grad_checkpoint={gc}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def step(batch):
        flat = [p for r in batch for p in r["pairs"]]
        ids, am = collate(flat, pad_id)
        ids, am = ids.to(dev), am.to(dev)
        logits = model(input_ids=ids, attention_mask=am).logits.squeeze(-1)
        maxk = max(r["k"] for r in batch)
        S = torch.full((len(batch), maxk), -1e4, device=dev)
        tgt = torch.tensor([r["answer"] for r in batch], device=dev)
        i = 0
        for b, r in enumerate(batch):
            S[b, :r["k"]] = logits[i:i + r["k"]]
            i += r["k"]
        return F.cross_entropy(S, tgt), len(flat)

    if args.benchmark:
        model.train()
        order = list(range(len(data)))
        rng.shuffle(order)
        chunk = sorted(order[:window], key=plen.__getitem__)
        plan = batches_by_budget(chunk, data, plen, args.tok_budget, args.max_pairs)
        print(f"benchmark: {len(plan)} steps over a {len(chunk)}-row window, "
              f"{sum(len(b) for b in plan) / max(1, len(plan)):.1f} rows/step", flush=True)
        for s in range(args.benchmark):
            batch = [data[i] for i in plan[s % len(plan)]]
            t = time.time()
            loss, npairs = step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            dt = time.time() - t
            vram = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
            print(f"bench {s+1}/{args.benchmark} loss={loss.item():.4f} "
                  f"{npairs/dt:.1f} pairs/s vram={vram:.2f}GB -> "
                  f"{dt * 163840 / npairs / 60:.1f} min per 164k-pair epoch", flush=True)
        return

    assert args.out, "--out is required for training"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt_latest"
    total_pairs = sum(r["k"] for r in data)
    steps_per_epoch = max(1, len(data) // args.batch_rows)
    total_steps = int(steps_per_epoch * args.epochs)

    # Resumable: model + optimizer + rng + step, so an interrupted run (the operator's
    # GPU is shared and can be taken back at any moment) continues where it stopped.
    done = 0
    if (ck / "state.pt").exists():
        st = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        done = st["step"]
        rng.setstate(st["rng"])
        print(f"RESUMED from step {done}", flush=True)

    def save_ckpt(step):
        ck.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "rng": rng.getstate(), "args": vars(args)}, ck / "state.pt.tmp")
        (ck / "state.pt.tmp").replace(ck / "state.pt")

    start_step = done
    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps} start_step={start_step} "
          f"save_steps={args.save_steps}", flush=True)

    model.train()
    t0, seen, hit_deadline, skipped = time.time(), 0, False, 0
    order = list(range(len(data)))
    stop = False
    while done < total_steps and not stop:
        rng.shuffle(order)
        for w in range(0, len(order), window):
            chunk = sorted(order[w:w + window], key=plen.__getitem__)
            for rows_idx in batches_by_budget(chunk, data, plen, args.tok_budget, args.max_pairs):
                if done >= total_steps:
                    stop = True
                    break
                batch = [data[i] for i in rows_idx]
                lr = args.lr * min(1.0, (done + 1) / args.warmup) * (1 - done / total_steps)
                for g in opt.param_groups:
                    g["lr"] = lr
                loss, npairs = step(batch)
                if not torch.isfinite(loss):
                    # A single bad batch used to kill the whole run (it died at step 43 of
                    # the first full-corpus attempt). Skip it, count it, keep going - but
                    # report the count so a systematic problem cannot hide.
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    print(f"WARN non-finite loss at step {done} - skipped "
                          f"({skipped} so far, rows {rows_idx[:4]}...)", flush=True)
                    continue
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gnorm):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    print(f"WARN non-finite grad norm at step {done} - skipped "
                          f"({skipped} so far, rows {rows_idx[:4]}...)", flush=True)
                    continue
                opt.step()
                opt.zero_grad(set_to_none=True)
                done += 1
                seen += npairs
                if args.step_sleep:
                    # idle window for the operator's compositor and video decode
                    time.sleep(args.step_sleep)
                if done % 50 == 0 or done == start_step + 1:
                    dt = time.time() - t0
                    run = max(1, done - start_step)
                    vram = (f" vram {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
                            if dev == "cuda" else "")
                    print(f"step {done}/{total_steps} loss {loss.item():.4f} lr {lr:.2e} "
                          f"{dt / run:.2f}s/step {seen / dt:.0f} pairs/s{vram}", flush=True)
                if args.save_steps and done % args.save_steps == 0:
                    save_ckpt(done)
                if args.deadline and time.time() >= args.deadline:
                    hit_deadline = True
                    save_ckpt(done)
                    print(f"DEADLINE reached at step {done}/{total_steps}", flush=True)
                    stop = True
                    break
            if stop:
                break

    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    json.dump({"base": args.base, "params": n_par, "rows": len(data), "pairs_per_epoch": total_pairs,
               "steps_done": done, "steps_this_run": done - start_step, "epochs": args.epochs,
               "max_len": args.max_len, "batch_rows": args.batch_rows, "lr": args.lr,
               "seed": args.seed, "wall_s": time.time() - t0, "corpus": args.corpus,
               "hit_deadline": hit_deadline, "skipped_steps": skipped},
              open(out / "train_args.json", "w"), indent=1)
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
