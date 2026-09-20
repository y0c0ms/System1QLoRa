#!/usr/bin/env python3
"""Interactive-ish demo of the trained 4B decision scorer.

Loads base + LoRA, scores each option's ' LETTER' logprob after the standard
State/Question/Options/Answer: prompt, softmaxes, and prints the ranked choices
with confidence. Scenarios span in-domain (math, code), zero-shot tool-calling,
and novel decisions the model never trained on - to show what generalises.
"""

import json
import string
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "results/jevlike4b_lora"
LETTERS = string.ascii_uppercase

SCENARIOS = [
    {"tag": "math_topic (in-domain)",
     "state": "A right triangle has legs of length 5 and 12. Find the length of the hypotenuse.",
     "question": "Which area of mathematics does this problem belong to?",
     "options": ["Algebra", "Geometry", "Number Theory", "Probability", "Calculus"]},
    {"tag": "code_defect (in-domain)",
     "state": "void copy(char *dst, const char *src) {\n    while (*src) { *dst++ = *src++; }\n    *dst = '\\0';\n}",
     "question": "Does this function contain a security vulnerability?",
     "options": ["no", "yes"]},
    {"tag": "tool-calling (ZERO-SHOT - never trained)",
     "state": "What's the weather like in Tokyo right now, and will it rain tomorrow?",
     "question": "Which function should be called?",
     "options": ["stock.get_price", "weather.get_forecast", "calendar.add_event",
                 "translate.text", "None of the above"]},
    {"tag": "novel: debugging triage (out-of-domain)",
     "state": "A user reports: 'My Python script prints the right answer but then hangs "
              "forever and never exits. I have a few threads and a queue.'",
     "question": "What is the most likely cause?",
     "options": ["A syntax error", "Non-daemon threads blocking interpreter exit",
                 "Insufficient RAM", "A missing import"]},
    {"tag": "novel: relational choice (Jev reportedly fails these)",
     "state": "Event A: the Berlin Wall fell in 1989. Event B: the first iPhone was released in 2007.",
     "question": "Which event happened first?",
     "options": ["Event A", "Event B", "They happened in the same year"]},
    {"tag": "novel: commonsense safety",
     "state": "You receive an email: 'URGENT: your account is locked. Click http://paypa1-secure.ru "
              "and enter your password to restore access.'",
     "question": "What should you do?",
     "options": ["Click the link and enter the password", "Report it as phishing and delete it",
                 "Forward it to all your contacts", "Reply with your password"]},
]


def build_prompt(row):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(row["options"]))
    return ("State:\n%s\n\nQuestion:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (row["state"], row["question"], opts))


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.truncation_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    lid = {L: tok(" " + L, add_special_tokens=False)["input_ids"][0] for L in LETTERS}

    @torch.no_grad()
    def score(row):
        ids = tok(build_prompt(row), return_tensors="pt", truncation=True, max_length=1536)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        lg = model(**ids).logits[0, -1].float()
        raw = torch.tensor([lg[lid[LETTERS[i]]] for i in range(len(row["options"]))])
        return torch.softmax(raw, dim=-1).tolist()

    for s in SCENARIOS:
        probs = score(s)
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        pick = order[0]
        print("\n" + "=" * 72)
        print(f"[{s['tag']}]")
        print(f"State: {s['state'][:120]}{'...' if len(s['state'])>120 else ''}")
        print(f"Q: {s['question']}")
        print(f"  -> PICK: {s['options'][pick]}  ({probs[pick]*100:.1f}%)")
        for i in order:
            bar = "#" * int(round(probs[i] * 30))
            print(f"      {probs[i]*100:5.1f}%  {bar:<30} {s['options'][i]}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
