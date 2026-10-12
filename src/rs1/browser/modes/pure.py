"""Reranker-only browser decisions: the same help jev-ultrafast gives Jev, nothing more.

Each step the reranker gets what Jev gets: jev-ultrafast's own instructions (NEXT_ACTION, TARGET),
the goal, the page (URL, title, visible text), every element with its role and current value, and
the last 10 actions with whether they changed the page. Every possible operation is a candidate,
including WAIT/SCROLL and DONE/BLOCKED; the highest-scoring one is executed. No rules on top.

Field text: Jev's loop asks a text LLM; there is none here, so the reranker picks the value from
the phrases of the goal (every run of 1-6 consecutive words).

    AGENT=pure uv run rs1-bench     (or pick "Reranker only" in the app)
"""
import math
import os
import re
import time

from jev_ultrafast.model import action_space
from jev_ultrafast.questions import NEXT_ACTION, TARGET, TEXT_VALUE

from rs1.reranker import shared

MODEL = os.environ.get("RERANKER", "Qwen/Qwen3-Reranker-0.6B")
rr = shared(MODEL)

ACTION_INSTRUCTION = (
    "Judge whether the Document is the single best next browser operation for the user's goal. " + NEXT_ACTION
    + " " + TARGET)
TEXT_INSTRUCTION = "Judge whether the Document is exactly the text to enter in the field. " + TEXT_VALUE
OPERATIONS = {
    "CLICK": "Click", "TYPE_TEXT": "Type text into", "SELECT": "Select",
}
FINISH = {
    "DONE": "DONE: every requirement is visibly satisfied.",
    "BLOCKED": "BLOCKED: no supported operation can make progress.",
}
PAGE_TEXT_CHARS = 1500


def describe(op, a):
    label = a["label"]
    value = a.get("current_value", a.get("value", ""))
    state = [f'role {a["role"]}'] if a.get("role") else []
    state += [f'currently "{value}"'] if value and op != "SELECT" else []
    state += [f"{k}={a[k]}" for k in ("checked", "selected", "expanded") if k in a]
    return f'{OPERATIONS[op]} "{label}"' + (f" ({', '.join(state)})" if state else "")


def recent(history):
    steps = []
    for h in history[-10:]:
        text = f' = "{h["text"]}"' if h.get("text") else ""
        changed = "" if h.get("page_changed") is not False else " (no visible change)"
        steps.append(f'{h["kind"]} {h["action"]}{text}{changed}')
    return "; ".join(steps) or "nothing yet"


def choose(state, goal, history, offer_done=True):
    started = time.perf_counter()
    _, targets, controls = action_space(state["actions"])
    cands = []  # (operation, target index or None, action id, document)
    for op, group in targets.items():
        for index, a in group.items():
            cands.append((op, index, a["id"], describe(op, a)))
    for name, a in controls.items():
        cands.append((name, None, a["id"], f'{name}: {a["label"]}'))
    for name, doc in FINISH.items():
        if name != "DONE" or offer_done:  # without it, stopping is left to a separate done check
            cands.append((name, None, name, doc))
    text = " ".join(state.get("text", "").split())[:PAGE_TEXT_CHARS]
    query = (f"Goal: {goal}\nPage: {state['title']} ({state['url']})\nPage text: {text}\n"
             f"Recent actions: {recent(history)}")
    scores = rr.score(ACTION_INSTRUCTION, query, [c[3] for c in cands])
    best = max(range(len(cands)), key=scores.__getitem__)
    operation, target, choice, _ = cands[best]
    # Same answer shape as Jev's: probabilities over the chosen operation's targets.
    m = max(scores)
    weights = [math.exp((s - m) * 20) for s in scores]
    total = sum(weights)
    probs = {c[2]: w / total for c, w in zip(cands, weights)}
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
        "raw_answers": {"reason": f"top score (p={scores[best]:.2f})",
                        "ranked": [[round(s, 4), d[:140]] for s, d in ranked], "picked": cands[best][3][:140]},
        "model": f"{MODEL.split('/')[-1]} (local, reranker only)",
        "usage": {"candidates": len(cands)},
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": {"query": query, "candidates": len(cands)},
    }


def goal_phrases(goal, longest=6):
    words = re.findall(r"[^\s,.;:!?]+(?:[,.](?=\d)[^\s,.;:!?]+)*", goal)
    phrases = []
    for n in range(1, longest + 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            if phrase not in phrases:
                phrases.append(phrase)
    # "November 20, 2026" keeps its comma: add each run as it is written in the goal too.
    for m in re.finditer(r"\b[A-Z][a-z]+ \d{1,2}, \d{4}\b", goal):
        if m.group() not in phrases:
            phrases.append(m.group())
    return phrases


def field_text(context):
    started = time.perf_counter()
    field = context["field"]
    phrases = goal_phrases(context["goal"])
    query = (f"Goal: {context['goal']}\nField: \"{field.get('label')}\" (role {field.get('role')}, "
             f"currently \"{field.get('value') or ''}\")\nRecent actions: "
             + "; ".join(f'{h["action"]}' + (f' = "{h["text"]}"' if h.get("text") else "")
                         for h in context["recent_actions"]))
    scores = rr.score(TEXT_INSTRUCTION, query, phrases)
    best = max(range(len(phrases)), key=scores.__getitem__)
    return phrases[best], {"model": f"{MODEL.split('/')[-1]} (picks a phrase of the goal, p={scores[best]:.2f})",
                           "latency_ms": round((time.perf_counter() - started) * 1000), "usage": {}}
