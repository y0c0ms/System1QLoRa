#!/usr/bin/env python3
"""LoRA fine-tune with the score head attached at layer 8 (not the final layer).

Same protocol and helpers as system_one.py train (Batcher, buckets, encode,
fit_temperature, run_eval) but with Gemma3TextForSequenceClassificationHeadAtLayer
so the head reads hidden_states[8] instead of the final layer. Runs on CPU with
the same memory-lean config as the SNI fit. Checkpoints/resumes like it too.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

import upstream.system_one as s  # reuse helpers/protocol
from harness.layer8_head import Gemma3TextForSequenceClassificationHeadAtLayer as Head8


def load_data(args):
    if args.local_data_dir:
        from datasets import load_from_disk
        return load_from_disk(args.local_data_dir)
    from datasets import load_dataset
    return load_dataset(args.data_repo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="mlx-community/gemma-3-270m-bf16")
    ap.add_argument("--data-repo", default=None)
    ap.add_argument("--local-data-dir", default="data/sni_ds")
    ap.add_argument("--out-dir", default="results/head8_fit")
    ap.add_argument("--head-layer", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-options", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=48)
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--resume-adapter", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true",
                    help="load adapter and eval a split (no training); T from val")
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter-dir", default=None)
    args = ap.parse_args()

    device = torch.device("cpu")
    dd = load_data(args)
    if args.eval_only:
        tok = AutoTokenizer.from_pretrained(args.base_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = Head8.from_pretrained(args.base_model, num_labels=1)
        model.head_layer = args.head_layer
        model.config.pad_token_id = tok.pad_token_id
        model.config.eos_token_id = tok.eos_token_id
        model.config.get_text_config().pad_token_id = tok.pad_token_id
        model.config.get_text_config().num_labels = 1
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          bias="none", task_type=TaskType.SEQ_CLS,
                          target_modules=s.LORA_TARGETS, modules_to_save=["score"])
        model = get_peft_model(model, lcfg)
        model = PeftModel.from_pretrained(
            model, args.adapter_dir or args.out_dir)
        model.to(device).eval()
        print(f"eval-only: head_layer={args.head_layer} adapter={args.adapter_dir or args.out_dir}",
              flush=True)

        class Args:
            pass
        a = Args()
        a.max_len = args.max_len
        a.eval_batch = args.eval_batch
        a.eval_limit = args.eval_limit

        rows = list(dd[args.split])
        if args.eval_limit:
            rows = s.take_per_task(rows, args.eval_limit)
        if args.split != "val":
            val_rows = list(dd["val"])
            if args.eval_limit:
                val_rows = s.take_per_task(val_rows, args.eval_limit)
            _, v_rec, v_log = s.run_eval(model, tok, val_rows, a, device, tag="val")
            T, nll = s.fit_temperature(v_log, [int(r["answer_index"]) for r in val_rows])
            print(f"temperature {T:.3f} val_nll {nll:.4f}", flush=True)
        else:
            T = None
        m, rec, logits = s.run_eval(model, tok, rows, a, device, tag=args.split)
        if T is None:
            T, nll = s.fit_temperature(logits, [int(r["answer_index"]) for r in rows])
            print(f"temperature {T:.3f}", flush=True)
        print("UNCALIBRATED " + json.dumps(m), flush=True)
        print("CALIBRATED " + json.dumps(
            s.metrics_from_records(s.apply_temperature(rec, logits, T))), flush=True)
        return

    train_rows = list(dd["train"])
    val_rows = list(dd["val"])
    rng = random.Random(0)
    rng.shuffle(train_rows)
    if args.max_steps:
        pass
    print(f"head_layer={args.head_layer} train {len(train_rows)} val {len(val_rows)}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Head8.from_pretrained(args.base_model, num_labels=1)
    model.head_layer = args.head_layer  # set post-load - see layer8_head.py
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.get_text_config().pad_token_id = tok.pad_token_id
    model.config.get_text_config().num_labels = 1
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type=TaskType.SEQ_CLS,
                      target_modules=s.LORA_TARGETS, modules_to_save=["score"])
    model = get_peft_model(model, lcfg)
    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter)
        print("resumed adapter from", args.resume_adapter, flush=True)
    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_trainable} of {n_total}", flush=True)

    batcher = s.Batcher(tok, args.max_len, seed=0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(1, len(s.buckets(train_rows, args.batch_size, seed=0,
                                           max_options=args.max_options)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = int(0.03 * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min((st + 1) / max(1, warmup),
                            0.5 * (1 + math.cos(math.pi * min(1.0, st / total_steps)))))

    step = 0
    t_start = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_rows = list(train_rows)
        rng.shuffle(epoch_rows)
        for bi, k in s.buckets(epoch_rows, args.batch_size, seed=epoch,
                               max_options=args.max_options):
            rows = [epoch_rows[i] for i in bi]
            batch = batcher(rows, k)
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
            logits = out.logits.squeeze(-1).view(len(rows), k)
            loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t_start
                print(f"step {step}/{total_steps} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} elapsed {el:.0f}s "
                      f"steps/s {step/max(el,1e-9):.3f}", flush=True)
            if args.save_every and step % args.save_every == 0:
                os.makedirs(args.out_dir, exist_ok=True)
                model.save_pretrained(args.out_dir)
                print(f"checkpoint step {step} -> {args.out_dir}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    train_seconds = time.time() - t_start
    print(f"train done {train_seconds:.0f}s for {step} steps", flush=True)

    model.eval()

    class Args:
        pass
    a = Args()
    a.max_len = args.max_len
    a.eval_batch = args.eval_batch
    a.eval_limit = args.eval_limit

    v_rows = take_per_task_guard(val_rows, args.eval_limit)
    val_m, val_rec, val_logits = s.run_eval(model, tok, v_rows, a, device, tag="val")
    T, nll = s.fit_temperature(val_logits, [int(r["answer_index"]) for r in v_rows])
    print(f"temperature {T:.3f} val_nll {nll:.4f}", flush=True)
    val_cal = s.metrics_from_records(s.apply_temperature(val_rec, val_logits, T))
    print("VAL UNCALIBRATED " + json.dumps(val_m), flush=True)
    print("VAL CALIBRATED " + json.dumps(val_cal), flush=True)

    test_m = test_cal = None
    if args.eval_test:
        t_rows = take_per_task_guard(list(dd["test"]), args.eval_limit)
        test_m, test_rec, test_logits = s.run_eval(model, tok, t_rows, a, device, tag="test")
        test_cal = s.metrics_from_records(s.apply_temperature(test_rec, test_logits, T))
        print("TEST UNCALIBRATED " + json.dumps(test_m), flush=True)
        print("TEST CALIBRATED " + json.dumps(test_cal), flush=True)

    results = {"head_layer": args.head_layer, "base_model": args.base_model,
               "epochs": args.epochs, "steps": step,
               "train_seconds": train_seconds,
               "steps_per_sec": step / max(train_seconds, 1e-9),
               "n_trainable": n_trainable, "n_total": n_total,
               "max_len": args.max_len, "batch_size": args.batch_size,
               "max_options": args.max_options, "lora_r": args.lora_r,
               "lr": args.lr, "temperature": T, "val_nll": nll,
               "val_uncalibrated": val_m, "val_calibrated": val_cal,
               "test_uncalibrated": test_m, "test_calibrated": test_cal}
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("saved to", args.out_dir, flush=True)


def take_per_task_guard(rows, limit):
    if limit:
        return s.take_per_task(rows, limit)
    return rows


if __name__ == "__main__":
    main()