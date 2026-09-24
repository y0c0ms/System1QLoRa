#!/usr/bin/env python3
"""Unified decision trainer for the letter-logprob readout.

Loss = CE over the option-letter subspace at the answer position (exactly the
eval readout: logits of the K option letters at the last real token, softmax
over options), optionally mixed with soft-target KD. One script covers every
arm of the plan so arms differ only in the flag under test:

  --precision nf4|bf16   4-bit NF4 QLoRA base vs bf16 base      (H1a)
  --lora-r 16|64         adapter capacity                        (H1b)
  --base ...             Qwen3-0.6B vs Qwen3.5-0.8B              (H2)
  --permute-prob p       on-the-fly option-order shuffling       (H3)

Safety (a GPU ring reset crashed the desktop on 2026-09-23 during a smoke of
full fine-tuning with bitsandbytes 8-bit Adam; a 14.9 GB peak without gradient
checkpointing risked compositor starvation):
  * gradient checkpointing is ON by default;
  * a hard per-process VRAM cap (--vram-cap-gb) makes a runaway job fail itself
    instead of starving the desktop;
  * no bitsandbytes optimizers - plain torch AdamW only.

Padding: RIGHT padding + gathering the last real token via an index-tensor
logits_to_keep. Left padding gave NaN under fp32-weights+bf16-autocast (a pad
query whose every key is masked -> softmax over all -inf), and right padding is
also the correct choice for recurrent (Gated DeltaNet) layers, which would
otherwise ingest pad tokens before the real sequence.

Resumable (AGENTS.md rule 9): trainable params + optimizer + step + data cursor
+ RNG are saved to <out>/ckpt_latest every --save-steps and reloaded on start.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from eval_decider import text_field  # canonical field rendering; see eval_decider.text_field

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TARGETS = {
    "dense": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # Qwen3.5 hybrid: dense + Gated-DeltaNet projections, excluding the a/b gate
    # projections that set the recurrent decay (NaN suspects in the first smoke).
    "hybrid": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
               "in_proj_qkv", "in_proj_z", "out_proj"],
}


def build_prompt(state, question, options):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(options))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (text_field(state), text_field(question), opts))


def letter_ids(tok):
    ids = []
    for L in LETTERS:
        t = tok(" " + L, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            raise SystemExit(f"' {L}' is not a single token for this tokenizer: {t}")
        ids.append(t[0])
    return ids


def permute(row, rng):
    k = len(row["options"])
    order = list(range(k))
    rng.shuffle(order)
    r = dict(row)
    r["options"] = [row["options"][i] for i in order]
    r["answer_index"] = order.index(row["answer_index"])
    if row.get("teacher_logits") is not None:
        r["teacher_logits"] = [row["teacher_logits"][i] for i in order]
    return r


def last_token_logits(model, enc):
    """Logits at each row's last real token (right padding), LM head run only there."""
    last = enc["attention_mask"].sum(1) - 1                     # [B]
    uniq = torch.unique(last)                                   # sorted positions
    out = model(**enc, logits_to_keep=uniq).logits              # [B, U, V]
    col = torch.searchsorted(uniq, last)
    return out[torch.arange(out.shape[0], device=out.device), col]  # [B, V]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--corpus", required=True, help="structured jsonl: state/question/options/answer_index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", choices=["nf4", "bf16"], default="bf16")
    ap.add_argument("--mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--targets", choices=list(TARGETS), default="dense")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=None, help="default 2e-4 (lora) / 2e-5 (full)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--permute-prob", type=float, default=0.0)
    ap.add_argument("--kd-alpha", type=float, default=1.0, help="1.0 = hard-label CE only")
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--vram-cap-gb", type=float, default=8.0)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-rows", type=int, default=0, help="debug: use only N rows")
    ap.add_argument("--deadline", type=float, default=0.0,
                    help="unix time; once reached, stop cleanly (adapter + train_args saved)")
    args = ap.parse_args()
    if args.precision == "nf4" and args.mode == "full":
        raise SystemExit("nf4 + full fine-tuning is not a valid combination")
    lr = args.lr if args.lr is not None else (2e-4 if args.mode == "lora" else 2e-5)
    gc = not args.no_grad_checkpoint

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb / total), 0)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt_latest"

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.truncation_side = "left"      # keep the Options/Answer tail
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lid = torch.tensor(letter_ids(tok))

    if args.precision == "nf4":
        from peft import prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gc)
    else:
        # full: fp32 master weights + bf16 autocast. lora: bf16 frozen base (PEFT
        # keeps adapter weights in fp32 by default).
        dtype = torch.float32 if args.mode == "full" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=dtype, device_map={"": 0}, attn_implementation="sdpa")
        if gc:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if args.mode == "lora":
                model.enable_input_require_grads()
    model.config.use_cache = False

    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            target_modules=TARGETS[args.targets], bias="none", task_type="CAUSAL_LM"))
    else:
        for p in model.get_input_embeddings().parameters():
            p.requires_grad_(False)   # tied to the LM head in these models
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"precision={args.precision} mode={args.mode} r={args.lora_r} base={args.base} "
          f"trainable={n_tr:,}/{n_all:,} ({100 * n_tr / n_all:.2f}%) lr={lr} gc={gc} "
          f"vram_cap={args.vram_cap_gb}GB", flush=True)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.weight_decay)

    def lr_at(step):
        if step < args.warmup:
            return lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))

    rows = [json.loads(l) for l in open(args.corpus)]
    rows = [r for r in rows if 2 <= len(r["options"]) <= len(LETTERS)
            and 0 <= r["answer_index"] < len(r["options"])]
    if args.limit_rows:
        rows = rows[: args.limit_rows]
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    # Length bucketing: character length is a cheap, monotone proxy for token length.
    plen = [len(r["state"]) + len(r["question"]) + sum(len(o) for o in r["options"]) for r in rows]
    BUCKET_ROWS = args.batch * 32          # sort within windows of 32 batches
    pending = []                           # batches (lists of row indices) not yet consumed
    step, cur = 0, 0

    if (ck / "state.pt").exists():
        st = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["trainable"], strict=False)
        opt.load_state_dict(st["opt"])
        step, cur, order = st["step"], st["cur"], st["order"]
        pending = st.get("pending", [])
        rng.setstate(st["rng"])
        print(f"RESUMED from step {step}", flush=True)

    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}

    def save_ckpt():
        ck.mkdir(parents=True, exist_ok=True)
        sd = {n: t.detach().cpu() for n, t in model.state_dict().items() if n in trainable_names}
        torch.save({"trainable": sd, "opt": opt.state_dict(), "step": step, "cur": cur,
                    "order": order, "pending": pending, "rng": rng.getstate(), "args": vars(args)},
                   ck / "state.pt.tmp")
        (ck / "state.pt.tmp").replace(ck / "state.pt")

    def next_batch():
        nonlocal cur, order, pending
        if not pending:
            if cur + BUCKET_ROWS > len(order):
                rng.shuffle(order)
                cur = 0
            window = sorted(order[cur:cur + BUCKET_ROWS], key=plen.__getitem__)
            cur += BUCKET_ROWS
            pending = [window[i:i + args.batch] for i in range(0, len(window), args.batch)]
            pending = [b for b in pending if len(b) == args.batch]
            rng.shuffle(pending)              # batch order stays random across lengths
        b = [rows[i] for i in pending.pop()]
        if args.permute_prob > 0:
            b = [permute(r, rng) if rng.random() < args.permute_prob else r for r in b]
        return b

    use_autocast = args.mode == "full"
    dev = next(model.parameters()).device
    lid_dev = lid.to(dev)
    arangeL = torch.arange(len(LETTERS), device=dev)

    def batch_loss(batch):
        prompts = [build_prompt(r["state"], r["question"], r["options"]) for r in batch]
        enc = tok(prompts, return_tensors="pt", truncation=True, max_length=args.max_len,
                  padding=True, add_special_tokens=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            logits = last_token_logits(model, enc)                           # [B, V]
        L = logits.float()[:, lid_dev]                                        # [B, 26]
        K = torch.tensor([len(r["options"]) for r in batch], device=dev)
        L = L.masked_fill(arangeL[None, :] >= K[:, None], float("-inf"))
        ans = torch.tensor([r["answer_index"] for r in batch], device=dev)
        loss = F.cross_entropy(L, ans, reduction="none")
        if args.kd_alpha < 1.0:
            T = args.kd_temp
            kl = torch.zeros_like(loss)
            for b, r in enumerate(batch):
                if r.get("teacher_logits") is None:
                    continue
                k = len(r["options"])
                tl = torch.tensor(r["teacher_logits"], device=dev, dtype=torch.float32)
                kl[b] = F.kl_div(F.log_softmax(L[b, :k] / T, -1), F.softmax(tl / T, -1),
                                 reduction="sum") * T * T
            loss = args.kd_alpha * loss + (1 - args.kd_alpha) * kl
        return loss.mean(), enc["input_ids"].numel()

    model.train()
    t0, toks = time.time(), 0
    start_step = step
    WARM = 5                          # steps excluded from the steady-state s/step figure
    t_warm, hit_deadline = None, False
    while step < args.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.grad_accum):
            loss, nt = batch_loss(next_batch())
            (loss / args.grad_accum).backward()
            acc += loss.item() / args.grad_accum
            toks += nt
        if not math.isfinite(acc):
            raise SystemExit(f"non-finite loss at step {step}: {acc}")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
        if step - start_step == WARM:
            t_warm = time.time()
        if step % 10 == 0 or step == start_step + 1:
            dt = time.time() - t0
            done = step - start_step
            print(f"step {step}/{args.max_steps} loss {acc:.4f} lr {lr_at(step - 1):.2e} "
                  f"{dt / done:.2f}s/step tok/s {toks / dt:.0f} "
                  f"vram {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        if step % args.save_steps == 0 and step < args.max_steps:
            save_ckpt()
        if args.deadline and time.time() >= args.deadline:
            hit_deadline = True
            print(f"DEADLINE reached at step {step}/{args.max_steps}", flush=True)
            break
    t_end = time.time()
    run = step - start_step
    sps_steady = (t_end - t_warm) / (run - WARM) if t_warm is not None and run >= WARM + 5 else None

    if args.mode == "lora":
        model.save_pretrained(str(out))
    else:
        model.to(torch.bfloat16).save_pretrained(str(out), safe_serialization=True)
        tok.save_pretrained(str(out))
    json.dump({**vars(args), "lr": lr, "trainable_params": n_tr, "steps_done": step,
               "steps_this_run": run, "wall_s": time.time() - t0, "sps_steady": sps_steady,
               "hit_deadline": hit_deadline}, open(out / "train_args.json", "w"), indent=1)
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
