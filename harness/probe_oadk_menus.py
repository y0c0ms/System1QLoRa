#!/usr/bin/env python3
"""Probe how the System-One scorer behaves as the OADK tool menu is reshaped.

`test_oadk_tools.py` measures one thing: routing accuracy over a fixed 9-option menu.
This probe measures the *menu shape* itself, because that turned out to be the
variable that decides whether the scorer is usable at all:

  menu_shape       one request set, five menu shapes (full / small+None / small-None)
  out_of_scope     requests no tool can satisfy, against no-None phase menus
  capability_menu  "can any of these tools do this?" as a separate Y/N question
  capability_tool  "is <top-1 candidate> the right tool?" — per-tool confirmation
  all              everything above

Protocol is identical to the other harnesses: the prompt ends in "\\n\\nAnswer:",
each option's " LETTER" first-token logprob is read, argmax is the pick, and the
softmax over those logprobs is the reported confidence. CPU fp32, adapter `06b`.

    python harness/probe_oadk_menus.py --exp all
    python harness/probe_oadk_menus.py --exp capability_tool

Writes results/probe_oadk_menus.json.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = "yocoms/system1-qlora"
SUBFOLDER = "06b"
_local = ROOT / "models" / "Qwen3-0.6B" / "model.safetensors"
BASE = str(_local.parent) if _local.exists() else "Qwen/Qwen3-0.6B"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

MOD = "Module: Pedidos (OutSystems 11, screen Pedidos with layout block PedidosLayout). "

# Verbatim tool descriptions as the OADK router renders them.
TOOL_TEXT = {
    "render_screen": "render_screen(screen) — show a screen as it currently is in the model (no publish, works offline)",
    "audit": "audit(scope?: module|screen, screen?) — check the model for structural and security defects before anything is published",
    "build_screen": "build_screen(screen, archetype: login|list|form, entity?, listScreen?, fields?) — create a screen from a verified archetype instead of assembling widgets by hand",
    "entities": "entities(entity?) — see the data model before building over it — entity names, attributes and types",
    "create_entity": "create_entity(name, attributes?, references?) — add a database entity (attributes and references) to the data model",
    "edit_widget": "edit_widget(screen, widget, property, value, expression?) — change one property of an existing widget",
    "edit_css": "edit_css(marker, rules) — change the module stylesheet — where the theme's colours live (e.g. the navbar)",
    "save": "save(note?, allowInsecure?) — write the module to its .oml file, once it verifies and passes the security audit",
    "None": "None of the above — no tool in this toolset satisfies the request",
}
FULL = ["render_screen", "audit", "build_screen", "entities", "create_entity",
        "edit_widget", "edit_css", "save", "None"]
PHASE_MENUS = {
    "lifecycle": ["render_screen", "audit", "build_screen", "save"],
    "styling": ["edit_widget", "edit_css"],
    "data": ["entities", "create_entity"],
}

# Styling requests that the fixed 9-option menu already routes correctly — these are
# the controls: if a reshaped menu breaks *them*, the menu shape is the problem.
STYLING_CASES = [
    ("Recolour the navbar to #ff4fa0.", "edit_css"),
    ("Change the theme colour of the top bar.", "edit_css"),
    ("The submit button says 'Submit' — make it say 'Save'.", "edit_widget"),
    ("Set Visible=false on the Alert widget in Pedidos.", "edit_widget"),
]

# No tool in this toolset can satisfy these (documented gaps in the OADK skill).
OUT_OF_SCOPE = [
    "Publish the module to the platform.",
    "Deploy this to production.",
    "Add a button to the Pedidos screen.",
    "Wire the Save button to create the Order entity.",
    "Delete the Login screen.",
    "Make the app work offline on mobile.",
]

IN_SCOPE = [
    ("Recolour the navbar to #ff4fa0.", "edit_css"),
    ("Set Visible=false on the Alert widget in Pedidos.", "edit_widget"),
    ("Write the module to disk.", "save"),
    ("Run the security audit over the whole module before we publish.", "audit"),
    ("What attributes does the Order entity have?", "entities"),
    ("Create a login screen called LoginV2.", "build_screen"),
]

CAP_OPTIONS = ["Yes — one of these tools can carry out this request",
               "No — none of these tools can carry out this request"]
YN_OPTIONS = ["Yes", "No"]


def _adapter_snapshot_path(subfolder):
    local = ROOT / "models" / ("system1-qlora-" + subfolder)
    if (local / "adapter_model.safetensors").exists():
        return str(local)
    from huggingface_hub import snapshot_download
    return str(Path(snapshot_download(ADAPTER, allow_patterns=[subfolder + "/*"],
                                      local_files_only=True)) / subfolder)


def options_block(keys):
    return "\n".join("%s. %s" % (LETTERS[i], TOOL_TEXT[k]) for i, k in enumerate(keys))


def menu_prompt(request, keys):
    return ("State:\n%sUser request: %s\n\nQuestion:\nWhich tool should be called?\n\n"
            "Options:\n%s\n\nAnswer:" % (MOD, request, options_block(keys)))


def capability_menu_prompt(request):
    tools = "\n".join("- " + TOOL_TEXT[k] for k in FULL[:-1])
    opts = "\n".join("%s. %s" % (LETTERS[i], t) for i, t in enumerate(CAP_OPTIONS))
    return ("State:\n%sUser request: %s\n\nQuestion:\nCan any of the following tools carry out "
            "this request?\n\nAvailable tools:\n%s\n\nOptions:\n%s\n\nAnswer:"
            % (MOD, request, tools, opts))


def capability_tool_prompt(request, tool, phrasing="right_tool"):
    if phrasing == "right_tool":
        q = "Is %s the right tool to carry out this request?" % tool
    else:
        q = "Does %s do what this request asks for?" % tool
    opts = "\n".join("%s. %s" % (LETTERS[i], t) for i, t in enumerate(YN_OPTIONS))
    return ("State:\n%sUser request: %s\n\nTool under consideration:\n%s\n\nQuestion:\n%s\n\n"
            "Options:\n%s\n\nAnswer:" % (MOD, request, TOOL_TEXT[tool], q, opts))


class Scorer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.tok.truncation_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.float32, device_map="cpu", attn_implementation="eager")
        self.model = PeftModel.from_pretrained(model, _adapter_snapshot_path(SUBFOLDER))
        self.model.eval()
        self.lid = {L: self.tok(" " + L, add_special_tokens=False)["input_ids"][0]
                    for L in LETTERS}

    @torch.no_grad()
    def score(self, prompt, n_opts):
        ids = self.tok(prompt, add_special_tokens=True, return_tensors="pt",
                       truncation=True, max_length=1536)
        t = time.perf_counter()
        logits = self.model(**ids).logits[0, -1]
        dt = (time.perf_counter() - t) * 1000
        lp = torch.log_softmax(logits.float(), dim=-1)
        lg = [lp[self.lid[LETTERS[i]]].item() for i in range(n_opts)]
        mx = max(lg)
        e = [2.718281828 ** (x - mx) for x in lg]
        s = sum(e)
        return lg, [x / s for x in e], dt

    def pick(self, prompt, keys):
        lg, pr, dt = self.score(prompt, len(keys))
        i = max(range(len(lg)), key=lambda j: lg[j])
        return keys[i], pr[i], dt


def exp_menu_shape(sc, out):
    """Same requests, five menu shapes — isolates the menu shape from the request."""
    rows = []
    shapes = {
        "full_9_with_None": FULL,
        "styling_3_with_None": ["edit_widget", "edit_css", "None"],
        "styling_3_None_first": ["None", "edit_widget", "edit_css"],
        "styling_2_no_None": ["edit_widget", "edit_css"],
        "styling_2_reversed_no_None": ["edit_css", "edit_widget"],
    }
    print("== menu_shape: 4 styling requests x 5 menu shapes ==")
    for request, want in STYLING_CASES:
        row = {"request": request, "want": want, "shapes": {}}
        for name, keys in shapes.items():
            pick, conf, dt = sc.pick(menu_prompt(request, keys), keys)
            row["shapes"][name] = {"pick": pick, "conf": round(conf, 3),
                                   "correct": pick == want, "ms": round(dt)}
        print("  %s" % request)
        for name, r in row["shapes"].items():
            print("    [%s] %-26s -> %-12s conf=%.2f" %
                  ("ok  " if r["correct"] else "MISS", name, r["pick"], r["conf"]))
        rows.append(row)
    out["menu_shape"] = rows
    for name in shapes:
        n = sum(1 for r in rows if r["shapes"][name]["correct"])
        print("  TOTAL %-26s %d/%d" % (name, n, len(rows)))


def exp_out_of_scope(sc, out):
    """Out-of-scope requests against no-None phase menus: there is no correct option."""
    rows = []
    print("\n== out_of_scope: 6 unanswerable requests x 3 no-None phase menus ==")
    for request in OUT_OF_SCOPE:
        row = {"request": request, "menus": {}}
        for name, keys in PHASE_MENUS.items():
            pick, conf, dt = sc.pick(menu_prompt(request, keys), keys)
            row["menus"][name] = {"pick": pick, "conf": round(conf, 3), "ms": round(dt)}
        row["max_conf"] = max(m["conf"] for m in row["menus"].values())
        print("  %-52s max_conf=%.2f" % (request[:52], row["max_conf"]))
        for name, r in row["menus"].items():
            print("    %-10s -> %-14s conf=%.2f" % (name, r["pick"], r["conf"]))
        rows.append(row)
    out["out_of_scope"] = rows
    confs = [r["max_conf"] for r in rows]
    print("  max confidence over all decisions: %.2f (median %.2f)"
          % (max(confs), sorted(confs)[len(confs) // 2]))


def exp_capability_menu(sc, out):
    """Scope asked as its own Y/N question, with the tools as context, not options."""
    rows = []
    print("\n== capability_menu: 'can any of these tools do this?' ==")
    for label, cases in (("out_of_scope", [(r, None) for r in OUT_OF_SCOPE]),
                         ("in_scope", IN_SCOPE)):
        for request, _want in cases:
            pick, conf, dt = sc.pick(capability_menu_prompt(request), CAP_OPTIONS)
            said_yes = pick.startswith("Yes")
            correct = said_yes == (label == "in_scope")
            rows.append({"request": request, "expect": label, "pick": pick.split(" —")[0],
                         "said_yes": said_yes, "conf": round(conf, 3), "correct": correct,
                         "ms": round(dt)})
            print("  [%s] expect=%-11s pick=%-4s conf=%.2f | %s" %
                  ("ok  " if correct else "MISS", label, pick.split(" —")[0], conf, request[:48]))
    out["capability_menu"] = rows
    oos = [r for r in rows if r["expect"] == "out_of_scope"]
    ins = [r for r in rows if r["expect"] == "in_scope"]
    print("  caught %d/%d out-of-scope; in-scope answered 'No' %d/%d"
          % (sum(1 for r in oos if not r["said_yes"]), len(oos),
             sum(1 for r in ins if not r["said_yes"]), len(ins)))


def exp_capability_tool(sc, out):
    """Route first (no-None menu), then confirm the winner against its own description."""
    rows = []
    print("\n== capability_tool: route (full menu, no None) then confirm the winner ==")
    no_none_full = FULL[:-1]
    cases = [(r, None, "out_of_scope") for r in OUT_OF_SCOPE] + \
            [(r, w, "in_scope") for r, w in IN_SCOPE]
    for request, want, label in cases:
        cand, cconf, _ = sc.pick(menu_prompt(request, no_none_full), no_none_full)
        row = {"request": request, "expect": label, "want": want,
               "candidate": cand, "candidate_conf": round(cconf, 3)}
        for phrasing in ("right_tool", "does_it"):
            pick, conf, dt = sc.pick(capability_tool_prompt(request, cand, phrasing), YN_OPTIONS)
            said_yes = pick == "Yes"
            row[phrasing] = {"pick": pick, "conf": round(conf, 3),
                             "correct": said_yes == (label == "in_scope")}
        print("  %-48s cand=%-13s(%.2f) right_tool=%-3s(%.2f) does_it=%-3s(%.2f)" %
              (request[:48], cand, cconf, row["right_tool"]["pick"], row["right_tool"]["conf"],
               row["does_it"]["pick"], row["does_it"]["conf"]))
        rows.append(row)
    out["capability_tool"] = rows
    for phrasing in ("right_tool", "does_it"):
        n = sum(1 for r in rows if r[phrasing]["correct"])
        oos_ok = sum(1 for r in rows if r["expect"] == "out_of_scope" and r[phrasing]["correct"])
        print("  %-11s %d/%d correct (rejected %d/6 out-of-scope)"
              % (phrasing, n, len(rows), oos_ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["menu_shape", "out_of_scope", "capability_menu",
                             "capability_tool", "all"])
    args = ap.parse_args()

    sc = Scorer()
    print("base=%s adapter=%s/%s device=cpu\n" % (BASE, ADAPTER, SUBFOLDER), flush=True)
    dest = ROOT / "results" / "probe_oadk_menus.json"
    out = {}
    if dest.exists():
        try:
            with open(dest, encoding="utf-8") as fh:
                out = json.load(fh)
        except (OSError, ValueError):
            out = {}
    out.update({"base": BASE, "adapter": ADAPTER, "subfolder": SUBFOLDER, "device": "cpu"})
    if args.exp in ("menu_shape", "all"):
        exp_menu_shape(sc, out)
    if args.exp in ("out_of_scope", "all"):
        exp_out_of_scope(sc, out)
    if args.exp in ("capability_menu", "all"):
        exp_capability_menu(sc, out)
    if args.exp in ("capability_tool", "all"):
        exp_capability_tool(sc, out)

    dest = ROOT / "results" / "probe_oadk_menus.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nSAVED %s" % dest)


if __name__ == "__main__":
    main()
