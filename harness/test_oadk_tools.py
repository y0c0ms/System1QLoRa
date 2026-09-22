#!/usr/bin/env python3
"""Tool-routing test for the System-One 0.6B scorer on the OADK MCP toolset.

Task: given an OutSystems-11 editing session (state) and a user request, pick
the tool that satisfies it (question), or "None of the above" when the toolset
has no matching tool (publishing, logic wiring, adding widgets to an existing
screen — all documented gaps in the OADK skill).

Same protocol as hf_score.py / test_06b_cpu.py: prompt ends "\\n\\nAnswer:",
read each option's ' LETTER' first-token logprob, argmax = pick. Adapter vs
unmodified base, CPU fp32.
"""

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

torch.set_num_threads(max(1, torch.get_num_threads() - 2))


def _adapter_snapshot_path(subfolder):
    local = ROOT / "models" / ("system1-qlora-" + subfolder)
    if (local / "adapter_model.safetensors").exists():
        return str(local)
    from huggingface_hub import snapshot_download
    return str(Path(snapshot_download(ADAPTER, allow_patterns=[subfolder + "/*"],
                                      local_files_only=True)) / subfolder)


# The OADK MCP toolset (8 tools) plus the abstention sentinel.
TOOLS_SHORT = [
    "render_screen(screen) — show a screen's widget tree and HTML preview",
    "audit(scope?, screen?) — check the model for verify errors and security findings",
    "build_screen(screen, archetype: login|list|form, entity?, listScreen?, fields?) — compose a screen from an archetype",
    "entities(entity?) — list or describe the data model's entities and attributes",
    "create_entity(name, attributes?, references?) — create or refine a database entity",
    "edit_widget(screen, widget, property, value, expression?) — change one property of an existing widget",
    "edit_css(marker, rules) — replace a marked section of the module stylesheet",
    "save(note?, allowInsecure?) — write the module to its .oml file",
    "None of the above — no tool in this toolset satisfies the request",
]

# Verbatim first-line descriptions from the OADK agent prompt (production router view).
TOOLS_REAL = [
    "render_screen(screen) — show a screen as it currently is in the model (no publish, works offline)",
    "audit(scope?: module|screen, screen?) — check the model for structural and security defects before anything is published",
    "build_screen(screen, archetype: login|list|form, entity?, listScreen?, fields?) — create a screen from a verified archetype instead of assembling widgets by hand",
    "entities(entity?) — see the data model before building over it — entity names, attributes and types",
    "create_entity(name, attributes?, references?) — add a database entity (attributes and references) to the data model",
    "edit_widget(screen, widget, property, value, expression?) — change one property of an existing widget",
    "edit_css(marker, rules) — change the module stylesheet — where the theme's colours live (e.g. the navbar)",
    "save(note?, allowInsecure?) — write the module to its .oml file, once it verifies and passes the security audit",
    "None of the above — no tool in this toolset satisfies the request",
]

NONE = 8

MOD = "Module: Pedidos (OutSystems 11, screen Pedidos with layout block PedidosLayout). "

# (user request, expected tool index)
CASES = [
    # render_screen
    ("Show me what the Pedidos screen looks like right now.", 0),
    ("Render the Layout block on its own so I can inspect it.", 0),
    ("I just recoloured the navbar — show the screen again to confirm the change.", 0),
    ("Before I touch anything, I want to see the widget tree of the Login screen.", 0),
    ("Preview the current state of the GestaoPedidos screen.", 0),
    # audit
    ("Run the security audit over the whole module before we publish.", 1),
    ("Does the GestaoPedidos screen have any structural defects?", 1),
    ("Check the model for verify errors and raw model calls.", 1),
    ("Is the module clean enough to save, or are there findings?", 1),
    # build_screen
    ("Create a login screen called LoginV2.", 2),
    ("Scaffold a list screen over the Order entity.", 2),
    ("Make a create form for Customer.", 2),
    ("Build a form screen over Customer that returns to GestaoClientes after saving.", 2),
    ("Turn GestaoPedidos into a list screen over the Order entity.", 2),
    ("I need a sign-in page for the app, call it PortalLogin.", 2),
    # entities
    ("What attributes does the Order entity have?", 3),
    ("List the entities in this module.", 3),
    ("Describe the Customer entity in full.", 3),
    ("I need the data model before I build the form — what's there?", 3),
    # create_entity
    ("Add an entity Invoice with attributes Number:Text, Amount:Currency.", 4),
    ("Create a Customer entity with Name:Text mandatory and a reference to Company.", 4),
    ("Add a Phone attribute to the Customer entity.", 4),
    ("Add a foreign key from Order to Customer.", 4),
    ("Create an entity for LMS progress with a reference to Course.", 4),
    # edit_widget
    ("Set the Title of the header on Pedidos to 'Pedidos 2026'.", 5),
    ("Change the Login screen's Username field so it is mandatory.", 5),
    ("The submit button says 'Submit' — make it say 'Save'.", 5),
    ("Set Visible=false on the Alert widget in Pedidos.", 5),
    ("The TopHeaderBar on Pedidos shows 'Home' — change that label to 'Início'.", 5),
    # edit_css
    ("Recolour the navbar to #ff4fa0.", 6),
    ("The header bar is green — make it dark blue.", 6),
    ("Add a CSS rule styling .os-header with a new background.", 6),
    ("Change the theme colour of the top bar.", 6),
    ("The links in the header are hard to read; adjust their colour in the stylesheet.", 6),
    # save
    ("Write the module to disk.", 7),
    ("Persist my changes to the .oml file.", 7),
    ("Save anyway, I know about the security warning — I'm mid-refactor.", 7),
    ("Record a note 'navbar rebrand' and save the module.", 7),
    # abstention: documented gaps in the OADK toolset
    ("Publish the module to the platform.", NONE),
    ("Deploy this to production.", NONE),
    ("Add a button to the Pedidos screen.", NONE),
    ("Wire the Save button to create the Order entity.", NONE),
    ("Delete the Login screen.", NONE),
    ("Make the app work offline on mobile.", NONE),
]


def build_prompt(request, tool_lines):
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(tool_lines))
    return ("State:\n%sUser request: %s\n\nQuestion:\nWhich tool should be called?\n\n"
            "Options:\n%s\n\nAnswer:" % (MOD, request, opts))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["short", "real"], default="real",
                    help="which tool descriptions the router sees")
    args = ap.parse_args()
    TOOLS = TOOLS_SHORT if args.variant == "short" else TOOLS_REAL
    print(f"variant={args.variant}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    print("base loaded", flush=True)

    letter_id = {L: tok(" " + L, add_special_tokens=False)["input_ids"][0]
                 for L in LETTERS}

    @torch.no_grad()
    def score(m, prompt):
        ids = tok(prompt, add_special_tokens=True, return_tensors="pt",
                  truncation=True, max_length=1536)
        t = time.perf_counter()
        logits = m(**ids).logits[0, -1]
        dt = (time.perf_counter() - t) * 1000
        lp = torch.log_softmax(logits.float(), dim=-1)
        return [lp[letter_id[LETTERS[i]]].item() for i in range(len(TOOLS))], dt

    def softmax(lg):
        mx = max(lg)
        e = [2.718281828 ** (x - mx) for x in lg]
        s = sum(e)
        return [x / s for x in e]

    prompts = [build_prompt(r, TOOLS) for r, _ in CASES]
    base_logs = [score(model, p) for p in prompts]

    model = PeftModel.from_pretrained(model, _adapter_snapshot_path(SUBFOLDER))
    model.eval()
    print("adapter loaded", flush=True)

    n_ok = n_base_ok = 0
    lats = []
    rows = []
    per_tool = {}
    for i, ((req, want), (blg, _)) in enumerate(zip(CASES, base_logs)):
        lg, dt = score(model, prompts[i])
        lats.append(dt)
        probs = softmax(lg)
        pick = max(range(len(lg)), key=lambda i: lg[i])
        bpick = max(range(len(blg)), key=lambda i: blg[i])
        ok, bok = pick == want, bpick == want
        n_ok += ok
        n_base_ok += bok
        name = TOOLS[want].split("(")[0].split(" —")[0]
        st = per_tool.setdefault(name, [0, 0, 0.0])
        st[0] += ok
        st[1] += 1
        if ok:
            st[2] += probs[pick]
        print(f"[{'ok' if ok else 'MISS'}] want={name:14s} pick={TOOLS[pick].split('(')[0].split(' —')[0]:14s} "
              f"conf={probs[pick]:.2f} base={'ok' if bok else 'MISS'} {dt:.0f}ms | {req[:58]}",
              flush=True)
        rows.append({"request": req, "want": want, "pick": pick, "correct": bool(ok),
                     "probs": probs, "latency_ms": dt, "base_pick": bpick,
                     "base_correct": bool(bok)})

    print("\nper-tool:")
    for name, (ok, n, conf) in sorted(per_tool.items()):
        print(f"  {name:14s} {ok}/{n}  mean_conf_when_correct={conf/max(ok,1):.2f}")
    out = {"base": BASE, "adapter": ADAPTER, "subfolder": SUBFOLDER, "device": "cpu",
           "variant": args.variant, "tools": TOOLS, "n_total": len(CASES),
           "n_correct_adapter": n_ok,
           "n_correct_base": n_base_ok, "latency_ms_mean": sum(lats) / len(lats),
           "per_tool": {k: {"correct": v[0], "n": v[1]} for k, v in per_tool.items()},
           "cases": rows}
    dest = ROOT / "results" / f"oadk_tools_06b_{args.variant}.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nADAPTER {n_ok}/{len(CASES)}  BASE {n_base_ok}/{len(CASES)}  "
          f"mean {sum(lats)/len(lats):.0f} ms/decision (CPU fp32)")
    print("SAVED", dest)


if __name__ == "__main__":
    main()
