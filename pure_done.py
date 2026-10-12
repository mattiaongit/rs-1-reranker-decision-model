"""Reranker + done check: the reranker alone chooses every move (pure.py), and the only addition is
the rules version's done check (decide.done_probability) deciding when to stop.

Shows how far the reranker gets when the one thing it can't do, recognising the goal is reached,
is handled for it.
"""
import time

import decide
import pure

_last_pick = {}  # id(history) -> (label, reranker score, page title) of the last move, for "just opened it"


def choose(state, goal, history):
    started = time.perf_counter()
    if history:
        decide._debug.clear()
        p = decide.done_probability(state, goal, _last_pick.get(id(history)))
        if p >= decide.DONE_THRESHOLD:
            return {"choice": "DONE", "operation": "DONE", "target": None, "confidence": p,
                    "probabilities": {"DONE": 1.0}, "operation_probabilities": {"DONE": 1.0},
                    "target_probabilities": {}, "target_confidence": None,
                    "raw_answers": {"reason": f"done check: page shows the goal achieved (p={p:.2f})",
                                    "done_parts": dict(decide._debug.get("done_parts", {}))},
                    "model": f"{pure.MODEL.split('/')[-1]} (local, reranker + done check)", "usage": {},
                    "latency_ms": round((time.perf_counter() - started) * 1000), "request": {}}
    answer = pure.choose(state, goal, history, offer_done=False)
    label = next((a["label"] for a in state["actions"] if a["id"] == answer["choice"]), None)
    if label:
        _last_pick[id(history)] = (label, answer["raw_answers"]["ranked"][0][0], state["title"])
    answer["model"] = f"{pure.MODEL.split('/')[-1]} (local, reranker + done check)"
    return answer


field_text = pure.field_text
