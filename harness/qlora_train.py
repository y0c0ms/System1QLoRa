#!/usr/bin/env python3
"""QLoRA SFT of a 7-8B decision scorer on the mixed corpus (ROCm container).

Objective is the eval metric itself: after the baseline_logprob prompt ending in
"\n\nAnswer:", maximise P(' LETTER') for the correct option. Loss is computed on
the completion tokens only; the prompt is masked (-100).

Runs inside rocm/pytorch (host py3.14 has no ROCm torch wheels). Checkpoints every
--save-steps so a killed run resumes (AGENTS.md rule 9). VRAM budget 16 GiB:
4-bit NF4 + gradient checkpointing + paged_adamw_8bit + micro-batch 1.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]


def tokenize_factory(tok, max_len):
    def fn(ex):
        p = tok(ex["prompt"], add_special_tokens=True)["input_ids"]
        c = tok(ex["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        # keep ALL completion tokens; left-truncate the prompt to preserve the
        # Options/Answer: tail (the decision cue) when the state is long.
        room = max_len - len(c)
        if room < 1:
            room = 1
        p = p[-room:]
        ids = p + c
        labels = [-100] * len(p) + c[:]  # loss only on completion
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}
    return fn


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * n)
            lab.append(f["labels"] + [-100] * n)
            att.append(f["attention_mask"] + [0] * n)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--corpus", default=str(ROOT / "data" / "mixed" / "train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "jevlike7b_lora"))
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="per-device micro-batch; raise for small models to use the GPU")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.corpus, split="train")
    ds = ds.map(tokenize_factory(tok, args.max_len),
                remove_columns=ds.column_names, num_proc=4)

    total_steps = (args.max_steps if args.max_steps > 0
                   else int(len(ds) / args.grad_accum * args.epochs))
    warmup = max(10, int(0.03 * total_steps))

    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=warmup,
        bf16=True, logging_steps=10, save_steps=args.save_steps, save_total_limit=3,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", report_to=[], dataloader_num_workers=2)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=PadCollator(tok.pad_token_id))
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    json.dump(vars(args), open(os.path.join(args.out, "train_args.json"), "w"), indent=1)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
