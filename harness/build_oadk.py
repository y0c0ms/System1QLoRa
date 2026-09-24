#!/usr/bin/env python3
"""Build a clean OADK tool-routing benchmark in the standard
{state, question, options, answer_index} schema, so our letter-logprob scorer,
kev's pointer head, and Jev can all be scored on identical items.

Source: the OADK menu probe (harness/probe_oadk_menus.py) — an OutSystems agent
that exposes 8 MCP tools + "None of the above". Each user request routes to one
tool, or to None when no tool in the set can satisfy it (documented OADK gaps).
We always present the FULL 9-option menu (which contains None), so abstention is
always expressible and every row has a well-defined label. Option order is
shuffled per row (seeded) so the signal is the routing decision, not a position.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MOD = "Module: Pedidos (OutSystems 11, screen Pedidos with layout block PedidosLayout). "
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
FULL = list(TOOL_TEXT)

# (request, gold tool key). In-scope + styling controls route to a tool;
# out-of-scope requests route to None (no tool in the set fits — real OADK gaps).
CASES = [
    ("Recolour the navbar to #ff4fa0.", "edit_css"),
    ("Change the theme colour of the top bar.", "edit_css"),
    ("The submit button says 'Submit' — make it say 'Save'.", "edit_widget"),
    ("Set Visible=false on the Alert widget in Pedidos.", "edit_widget"),
    ("Write the module to disk.", "save"),
    ("Run the security audit over the whole module before we publish.", "audit"),
    ("What attributes does the Order entity have?", "entities"),
    ("Create a login screen called LoginV2.", "build_screen"),
    ("Show me the Pedidos screen as it looks now.", "render_screen"),
    ("Add a Status attribute to the Order entity.", "create_entity"),
    # out-of-scope -> None (documented gaps: no tool can do these)
    ("Publish the module to the platform.", "None"),
    ("Deploy this to production.", "None"),
    ("Add a button to the Pedidos screen.", "None"),
    ("Wire the Save button to create the Order entity.", "None"),
    ("Delete the Login screen.", "None"),
    ("Make the app work offline on mobile.", "None"),
]


def build_row(request, gold, rng):
    keys = FULL[:]
    rng.shuffle(keys)
    options = [TOOL_TEXT[k] for k in keys]
    return {
        "task": "oadk_tool_routing",
        "state": MOD + "User request: " + request,
        "question": "Which tool should be called?",
        "options": options,
        "answer_index": keys.index(gold),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "oadk"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [build_row(req, gold, rng) for req, gold in CASES]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # tiny benchmark: score all as test; a shuffled copy serves temperature val
    json.dump(rows, open(out / "test.json", "w"), indent=1)
    vrows = [build_row(req, gold, random.Random(args.seed + 1)) for req, gold in CASES]
    json.dump(vrows, open(out / "val.json", "w"), indent=1)
    n_none = sum(1 for _, g in CASES if g == "None")
    print(f"wrote {len(rows)} OADK rows ({n_none} abstain / {len(rows) - n_none} route) -> {out}")


if __name__ == "__main__":
    main()
