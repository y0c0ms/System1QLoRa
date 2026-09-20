#!/usr/bin/env python3
"""Post a set of hard/adversarial decisions to the running 4B server and dump
results/demo_hard_results.json for the Astro results page."""

import json
import urllib.request

URL = __import__("os").environ.get("MODEL_SERVER", "http://127.0.0.1:8900/score")

CASES = [
    {"tag": "Numeric trap (LLMs often fail)", "expect": 1,
     "state": "Compare the two decimal numbers 9.11 and 9.9.",
     "question": "Which number is larger?",
     "options": ["9.11", "9.9"]},
    {"tag": "Weight trap", "expect": 2,
     "state": "You have 1 kilogram of feathers and 1 kilogram of steel.",
     "question": "Which weighs more?",
     "options": ["The feathers", "The steel", "They weigh the same"]},
    {"tag": "Unit trap", "expect": 2,
     "state": "Object A weighs 100 grams. Object B weighs 0.1 kilograms.",
     "question": "Which object is heavier?",
     "options": ["Object A", "Object B", "They are equal"]},
    {"tag": "Subtle off-by-one overflow", "expect": 1,
     "state": "char buf[8];\nvoid f(const char *s) {\n    for (int i = 0; i <= 8; i++) buf[i] = s[i];\n}",
     "question": "Does this function contain a security vulnerability?",
     "options": ["no", "yes"]},
    {"tag": "Safe code (should say no)", "expect": 0,
     "state": "int add(int a, int b) {\n    long r = (long)a + b;\n    if (r > INT_MAX || r < INT_MIN) return 0;\n    return (int)r;\n}",
     "question": "Does this function contain a security vulnerability?",
     "options": ["no", "yes"]},
    {"tag": "Transitive spatial reasoning", "expect": 2,
     "state": "Town A is north of Town B. Town B is north of Town C. Town C is north of Town D.",
     "question": "Which town is the southernmost?",
     "options": ["Town A", "Town B", "Town D", "Town C"]},
    {"tag": "Negation in entailment", "expect": 1,
     "state": "Premise: The scientist did not confirm that the experiment succeeded.",
     "question": "Does the premise entail that the experiment definitely succeeded?",
     "options": ["Yes, it is entailed", "No, it is not entailed", "It is a contradiction of itself"]},
    {"tag": "Sarcasm / true sentiment", "expect": 1,
     "state": "Review: 'Oh fantastic, another three-hour delay. Exactly how I wanted to spend my evening. Best airline ever.'",
     "question": "What is the true sentiment of this review?",
     "options": ["Positive", "Negative", "Neutral"]},
    {"tag": "Tool-calling: no tool fits (zero-shot)", "expect": 3,
     "state": "Please write me a haiku about the ocean.",
     "question": "Which function should be called?",
     "options": ["weather.get_forecast", "email.send", "calculator.evaluate", "None of the above"]},
    {"tag": "Temporal with distractors", "expect": 2,
     "state": "The Apollo 11 moon landing was 1969. The fall of the Berlin Wall was 1989. "
              "The invention of the telephone was 1876. The launch of Sputnik was 1957.",
     "question": "Which of these happened earliest?",
     "options": ["Apollo 11 moon landing", "Fall of the Berlin Wall",
                 "Invention of the telephone", "Launch of Sputnik"]},
]


def post(case):
    data = json.dumps({k: case[k] for k in ("state", "question", "options")}).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    out = []
    for c in CASES:
        res = post(c)
        res["tag"] = c["tag"]
        res["state"] = c["state"]
        res["question"] = c["question"]
        res["expect"] = c["expect"]
        res["correct"] = (res["pick"] == c["expect"])
        out.append(res)
        mark = "OK " if res["correct"] else "MISS"
        print(f"[{mark}] {c['tag']}: pick='{res['options'][res['pick']]}' "
              f"({res['probs'][res['pick']]*100:.0f}%) expected='{c['options'][c['expect']]}'")
    n_ok = sum(r["correct"] for r in out)
    print(f"\n{n_ok}/{len(out)} correct")
    json.dump({"n_correct": n_ok, "n_total": len(out), "cases": out},
              open("results/demo_hard_results.json", "w"), indent=1)
    print("wrote results/demo_hard_results.json")


if __name__ == "__main__":
    main()
