"""Assisted mode: code prepares the question, the reranker makes every choice.

Same inputs as pure.py (jev-ultrafast's instructions, goal, page, recent actions), and the
highest-scoring candidate is always executed. Code only shapes the candidate list where the
reranker alone was seen to fail, and never overrides its pick:

  1. Dropdowns: a closed dropdown is described with the choices hidden inside it, read from the
     page without opening it ("Change ticket type (now Round trip; choices: One way, Multi-city)").
  2. Typing: each field is offered once per value named in the goal, as a complete action
     ('Type "Zurich" into "Where from?" (the goal says "from Zurich")'), so the reranker picks
     field and value together instead of a chunk of the goal.
  3. Progress: actions already done from this same page (so no going back and forth), fields
     already holding a value from the goal, and controls the agent already set are not offered again.
  4. Done: its own reranker question each step, "is this page (title + address) where the goal is
     achieved?", instead of one more option competing with links.
"""
import json
import math
import re
import time

from jev_ultrafast.browser import Browser
from jev_ultrafast.model import action_space
from jev_ultrafast.questions import NEXT_ACTION

import decide  # shares the loaded reranker; reuses its text helpers and the element-context patch
from decide import goal_values, holds_goal_value, norm

rr = decide.rr
DONE_INSTRUCTION = ("Given a user's goal for a website and the page the browser is on now, judge whether "
                    "this page is where the goal is fully achieved.")
INSTRUCTION = "Judge whether the Document is the single best next browser operation for the user's goal. " + NEXT_ACTION
OPERATIONS = {"CLICK": "Click", "TYPE_TEXT": "Type into", "SELECT": "Select"}
PAGE_TEXT_CHARS = 1500
_next_text = {}  # field label -> value chosen with it
_chosen = {}  # id(history) -> [(steps so far, page, operation, label)] for every decision made

# Choices hidden inside a closed dropdown, read from the page (aria-controls / aria-owns / a
# <select>'s options / a listbox next to it). Reading them clicks nothing.
HIDDEN_OPTIONS_JS = """(nodes => nodes.map(id => {
  const e = window.__jevFast?.nodes.get(id); if (!e) return [];
  const ids = (e.getAttribute('aria-controls') || e.getAttribute('aria-owns') || '').split(/\\s+/).filter(Boolean);
  let boxes = ids.map(i => document.getElementById(i)).filter(Boolean);
  if (e.tagName === 'SELECT') return [...e.options].map(o => o.textContent.trim());
  if (!boxes.length) { const s = e.closest('div,span,li')?.parentElement; if (s) boxes = [...s.querySelectorAll('[role=listbox]')]; }
  const out = [];
  for (const b of boxes) for (const o of b.querySelectorAll('[role=option],option,li'))
    { const t = (o.getAttribute('aria-label') || o.textContent || '').trim(); if (t && t.length < 60 && !out.includes(t)) out.push(t); }
  return out.slice(0, 8);
}))(%s)"""

_observe = Browser.observe


def observe(self, *a, **k):
    page = _observe(self, *a, **k)
    closed = sorted({x["node"] for x in page["actions"]
                     if x.get("role") == "combobox" and str(x.get("expanded")).lower() == "false" and "node" in x})
    if closed:
        try:
            found = self.evaluate(HIDDEN_OPTIONS_JS % json.dumps(closed)) or []
            hidden = dict(zip(closed, found))
            for x in page["actions"]:
                if hidden.get(x.get("node")):
                    x["hidden_options"] = hidden[x["node"]]
        except Exception:
            pass  # a description hint; never fail an observation over it
    return page


Browser.observe = observe


def describe(op, a, value=None, said=None):
    label = a["label"].split(" → ")[0] if op != "SELECT" else a["label"]
    current = a.get("current_value", a.get("value", ""))
    state = [f'now "{current[:60]}"'] if current and op != "SELECT" else []
    state += [f"{k}={a[k]}" for k in ("checked", "selected") if k in a]
    others = [o for o in a.get("hidden_options", []) if norm(o) != norm(current)]
    if others:
        state.append("choices inside: " + ", ".join(others[:6]))
    ctx = " ".join(a.get("context", "").replace(label.strip(), " … ").split())
    tail = (f" ({'; '.join(state)})" if state else "") + (f' — in: "{ctx[:90]}"' if ctx and norm(ctx) else "")
    if op == "TYPE_TEXT":
        return f'Type "{value}" into "{label}"' + (f' (the goal says "{said}")' if said and said != value else "") + tail
    return f'{OPERATIONS[op]} "{label}"' + tail


def recent(history):
    out = []
    for h in history[-10:]:
        text = f' = "{h["text"]}"' if h.get("text") else ""
        out.append(f'{h["kind"]} {h["action"]}{text}' + (" (no visible change)" if h.get("page_changed") is False else ""))
    return "; ".join(out) or "nothing yet"


def choose(state, goal, history):
    started = time.perf_counter()
    _, targets, controls = action_space(state["actions"])
    values = goal_values(goal)
    typed = {(h["action"], norm(h["text"])) for h in history if h.get("text")}
    last = history[-1] if history else None
    no_change = {(h["operation"], h["action"]) for h in history[-4:] if h.get("page_changed") is False}
    set_by_us = {h["action"].split(" → ")[0] for h in history if h["kind"] == "select"} | \
                {h["action"] for h in history if h["kind"] == "click"}
    page = (state["url"], state["title"])
    # Actions executed earlier from this very page: a decision counts once the next history entry is it.
    done_here = {(op, label) for n, pg, op, label in _chosen.get(id(history), [])
                 if pg == page and n < len(history) and history[n]["action"] == label}
    cands, removed = [], []  # (operation, target, action id, document, value)

    def offer(op, index, a, doc, value=None, why_not=None):
        (removed.append((why_not, doc[:80])) if why_not else cands.append((op, index, a["id"], doc, value)))

    for op, group in targets.items():
        for index, a in group.items():
            key = (op, a["label"])
            again = last and key == (last["operation"], last["action"])
            if op == "TYPE_TEXT":
                label = a["label"]
                held = holds_goal_value(a.get("value", ""), goal)
                for value, said in (values or {"": None}).items():
                    if not value:
                        continue
                    why = "field already holds a value from the goal" if held else \
                        "this value was already typed here" if (label, norm(value)) in typed else None
                    offer(op, index, a, describe(op, a, value, said), value, why)
                continue
            role = a.get("role", "")
            why = None
            if again or key in no_change or key in done_here:
                why = "just done" if again else "did nothing before" if key in no_change else "already done from this page"
            elif op == "CLICK" and role in ("textbox", "searchbox", "combobox") and holds_goal_value(a.get("value", ""), goal):
                why = "field already holds a value from the goal"
            elif op == "SELECT" and a["label"].split(" → ")[0] in set_by_us:
                why = "already set by the agent"
            elif op == "CLICK" and role in decide.TOGGLES and a["label"] in set_by_us:
                why = "already set by the agent"
            offer(op, index, a, describe(op, a), None, why)
    for name, a in controls.items():
        if name == "WAIT" and last and last["kind"] == "wait" and len(history) > 1 and history[-2]["kind"] == "wait":
            removed.append(("waited twice already", name))
            continue
        cands.append((name, None, a["id"], f'{name}: {a["label"]}', None))
    cands.append(("BLOCKED", None, "BLOCKED", "BLOCKED: no supported operation can make progress.", None))

    # Done is its own reranker question (does this page fulfil the goal?), asked before choosing an action.
    if history:
        url = state["url"].split("?")[0]
        p_done = rr.score(DONE_INSTRUCTION, f"Goal: {goal}", [f'Page title: {state["title"]}\nURL: {url}'],
                          max_doc_tokens=96)[0]
        if p_done >= 0.5:
            return finish(p_done, started, len(cands))

    text = " ".join(state.get("text", "").split())[:PAGE_TEXT_CHARS]
    query = (f"Goal: {goal}\nPage: {state['title']} ({state['url'].split('?')[0]})\nPage text: {text}\n"
             f"Recent actions: {recent(history)}")
    scores = rr.score(INSTRUCTION, query, [c[3] for c in cands], max_doc_tokens=80)
    best = max(range(len(cands)), key=scores.__getitem__)
    operation, target, choice, doc, value = cands[best]
    label_of = {a["id"]: a["label"] for g in targets.values() for a in g.values()}
    _chosen.setdefault(id(history), []).append((len(history), page, operation, label_of.get(choice, choice)))
    if value:
        label = next(a["label"] for a in targets["TYPE_TEXT"].values() if a["id"] == choice)
        _next_text[label] = value
    m = max(scores)
    weights = [math.exp((s - m) * 20) for s in scores]
    total = sum(weights)
    probs = {}
    for c, w in zip(cands, weights):
        probs[c[2]] = max(probs.get(c[2], 0), w / total)
    op_probs = {}
    for c in cands:
        op_probs[c[0]] = op_probs.get(c[0], 0) + probs[c[2]]
    ranked = sorted(zip(scores, (c[3] for c in cands)), reverse=True)[:12]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": probs[choice],
        "probabilities": {c[2]: probs[c[2]] for c in cands if c[0] == operation},
        "operation_probabilities": op_probs,
        "target_probabilities": {c[1]: probs[c[2]] for c in cands if c[0] == operation and c[1]},
        "target_confidence": probs[choice] if target else None,
        "raw_answers": {"reason": f"reranker's top choice (p={scores[best]:.2f})",
                        "ranked": [[round(s, 4), d[:140]] for s, d in ranked],
                        "picked": doc[:140],
                        "removed": removed[:20]},
        "model": f"{decide.MODEL.split('/')[-1]} (local, assisted)",
        "usage": {"candidates": len(cands)},
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": {"query": query, "candidates": len(cands)},
    }


def finish(p_done, started, n):
    return {"choice": "DONE", "operation": "DONE", "target": None, "confidence": p_done,
            "probabilities": {"DONE": 1.0}, "operation_probabilities": {"DONE": 1.0}, "target_probabilities": {},
            "target_confidence": None, "raw_answers": {"reason": f"reranker: this page fulfils the goal (p={p_done:.2f})"},
            "model": f"{decide.MODEL.split('/')[-1]} (local, assisted)", "usage": {"candidates": n},
            "latency_ms": round((time.perf_counter() - started) * 1000), "request": {}}


def field_text(context):
    label = context["field"].get("label") or ""
    value = _next_text.pop(label, None)
    if not value:
        raise ValueError("No value was chosen with this field; nothing typed.")
    return value, {"model": "reranker chose this value with the field", "latency_ms": 0, "usage": {}}
