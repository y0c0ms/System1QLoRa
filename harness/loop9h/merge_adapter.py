#!/usr/bin/env python3
"""Merge a LoRA adapter into its base on CPU and save a bf16 HF checkpoint
(input for llama.cpp's convert_hf_to_gguf.py)."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    m = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": "cpu"})
    m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
    m.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    print("MERGED", a.out)


if __name__ == "__main__":
    main()
