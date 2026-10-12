"""Browser decisions with a local reranker: jev-ultrafast's choose() and field_text(), no LLM.

Per step:
  0. after typing, if suggestions appeared, pick the matching suggestion
  1. requested inputs that aren't satisfied yet, in page order (forms fill top to bottom)
  2. after typing with no suggestions, submit
  3. otherwise the most relevant element (reranker over goal + steps taken + element context)
  Done is a separate check on the page (title/URL and text), allowed only when no requested
  input is pending. Wait/scroll are fallbacks when nothing on the page is relevant.

"Requested" is decided generically: the value appears in the goal (accent/hyphen-insensitive
word match), or for a text field, the reranker finds a convincing value for it in the goal.
"""
import json
import math
import os
import re
import time
import unicodedata

from jev_ultrafast.browser import Browser
from jev_ultrafast.model import action_space

from reranker import shared
from shortlist import Shortlist

MODEL = os.environ.get("RERANKER", "Qwen/Qwen3-Reranker-0.6B")
rr = shared(MODEL)
sl = Shortlist()
_timing = []  # (instruction head, n docs, ms) for the decision in progress
_debug = {}
_score = rr.score


def _timed_score(instruction, query, docs, **k):
    t = time.perf_counter()
    out = _score(instruction, query, docs, **k)
    _timing.append((instruction.split()[1:4], len(docs), round((time.perf_counter() - t) * 1000)))
    return out


rr.score = _timed_score

ACTION_INSTRUCTION = (
    "Given a user's goal for a web page and the steps already taken, judge whether the Document "
    "is the single best next browser action to make progress toward the goal."
)
OBSTACLE_INSTRUCTION = (
    "Judge whether the Document is the browser action that dismisses an obstacle covering the page, "
    "such as a cookie or privacy consent dialog, a popup or a banner, so the user can continue."
)
OBSTACLE_QUERY = "Get past the cookie consent, privacy notice or popup to reach the actual website."
DONE_INSTRUCTION = (
    "Given a user's goal and the current web page, judge whether the page shows that the goal "
    "has been fully achieved."
)
TEXT_INSTRUCTION = (
    "Given a text field on a web page, judge whether the Document (a phrase from the user's request) "
    "states the value that belongs in that field."
)
SUBMIT_INSTRUCTION = "Judge whether the Document is the button or link that submits or runs the search for the field."

ELEMENT_FLOOR = 0.02      # below this nothing on the page looks relevant to the goal
FIELD_WANTED = 0.5        # a goal phrase this convincing for a text field means the goal asks to fill it
SLOT_MIN = 0.6            # slot_score needed to put a goal value into a text field
# Generic "the site failed" messages; after one, the click that led there is tried again once.
RETRY_RE = re.compile(r"^\s*(?:retry|reload|try again|refresh)\b", re.I)
_error_retries = {}  # id(history) -> steps at which an error page was retried
MAX_ERROR_RETRIES = 5
ERROR_RE = re.compile(r"\b(?:oops|something went wrong|an error (?:has )?occurred|please try again|try again later)\b", re.I)
DONE_THRESHOLD = 0.5
SHORTLIST_OVER = 60       # pages with more candidates than this get the cheap embedding pass first
SHORTLIST_K = 40
TOGGLES = {"checkbox", "radio", "switch", "menuitemcheckbox", "menuitemradio"}
OPTION_ROLES = {"option", "menuitem", "gridcell"}
STOP = set("a an the to for in on of and or with from by at into then find open use show search stop do not "
           "is are be this that it its as one when only about".split())


# ---------- text helpers ----------

def norm(s):
    s = re.sub(r"['’ʼ`´]", "", str(s or ""))  # "Gödel’s" and "Gödel's" both -> "godels"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def in_goal(text, goal):
    """Whole-word, accent/hyphen-insensitive containment: "Zürich" in "...from Zurich to...", "One way" in "one-way"."""
    t = norm(text)
    return len(t) >= 2 and f" {t} " in f" {norm(goal)} "


MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DATE_RE = re.compile(rf"\b(?:(?:{MONTHS})\.? \d{{1,2}}(?:st|nd|rd|th)?(?:,? \d{{4}})?|\d{{1,2}} (?:{MONTHS})(?: \d{{4}})?|"
                     rf"\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})\b", re.I)


def goal_values(goal):
    """Things people type into fields: dates, quoted strings, and names (capitalised runs, extended
    through following non-filler words: "Gödel’s incompleteness theorems"). Each comes with the word
    before it, which carries its role: ("from Zurich", "Zurich"), ("to London", "London")."""
    out = {}

    def add(start, end):
        value = goal[start:end].strip(" .,;:")
        if not value or value.lower() in STOP:
            return
        before = [w for w in re.findall(r"[\w’'-]+", goal[:start]) if w.lower() not in ("the", "a", "an")][-1:]
        out.setdefault(value, (" ".join(before + [value])))

    for m in DATE_RE.finditer(goal):
        add(m.start(), m.end())
    for m in re.finditer(r"[\"“'‘]([^\"”'’]{2,60})[\"”'’]", goal):
        add(m.start(1), m.end(1))
    # topics, even lowercase: "about photosynthesis", "called ..."
    for m in re.finditer(r"\b(?:about|called|named|titled|on the topic of)\s+(?:the\s+|a\s+|an\s+)?"
                         r"([\w’'-]+(?:\s+(?!(?:and|then|with|for|in|on|to|from|by)\b)[\w’'-]+){0,4})", goal, re.I):
        add(m.start(1), m.end(1))
    tokens = list(re.finditer(r"[\w’'-]+", goal))
    i = 0
    while i < len(tokens):
        w = tokens[i].group()
        sentence_start = i == 0 or goal[tokens[i - 1].end():tokens[i].start()].strip() in (".", "!", "?")
        if w[0].isupper() and not (sentence_start and w.lower() in STOP) and not DATE_RE.match(goal[tokens[i].start():]):
            j = i + 1
            while j < len(tokens) and j - i < 5 and tokens[j].group().lower() not in STOP \
                    and goal[tokens[j - 1].end():tokens[j].start()].strip() == "":
                j += 1
            add(tokens[i].start(), tokens[j - 1].end())
            i = j
        else:
            i += 1
    return out  # value -> value with its preceding word


def goal_phrases(goal, max_words=5):
    """Fallback when the goal names nothing: contiguous phrases not starting/ending on filler words."""
    words = [w.strip(".,;:") for w in re.findall(r"[\w’'.,\-]+", goal)]
    words = [w for w in words if w]
    out = []
    for i in range(len(words)):
        for j in range(i + 1, min(i + max_words, len(words)) + 1):
            span = words[i:j]
            if span[0].lower() in STOP or span[-1].lower() in STOP:
                continue
            phrase = " ".join(span)
            if phrase not in out:
                out.append(phrase)
    return out or [goal]


_field_cache = {}


DATE_LABEL = re.compile(r"date|depart|return|check ?in|check ?out|arriv|when|day|from date|to date")
PLACE_LABEL = re.compile(r"where|destination|city|location|place|origin|address|going|leaving|airport|station|from|to\b")
SEARCH_LABEL = re.compile(r"search|query|find|keyword|look ?up|what")
PLACE_WORDS = {"in", "to", "from", "at", "near", "into"}
TOPIC_WORDS = {"about", "for", "on", "called", "named", "titled"}


def slot_score(label, value, context):
    """How well a goal value fits a text field, from generic form conventions. The reranker is good
    at choosing actions but can't do this (it scores "Type 'September 20' into 'Where from?'" 0.95)."""
    lab = norm(label)
    words = context.split()
    before = norm(words[0]) if len(words) > 1 else ""
    is_date = bool(DATE_RE.fullmatch(value.strip()))
    score = 0.0
    if before and before in lab.split():
        score += 1.0                                    # "from Zurich" -> "Where from?"
    if is_date:
        score += 0.8 if DATE_LABEL.search(lab) else -0.5
    elif DATE_LABEL.search(lab) and not PLACE_LABEL.search(lab):
        score -= 0.5                                    # a name doesn't go in a date field
    if not is_date and before in PLACE_WORDS and PLACE_LABEL.search(lab):
        score += 0.7                                    # "in Lisbon" -> "Destination"
    if not is_date and SEARCH_LABEL.search(lab):
        score += 0.5 + 0.3 * (before in TOPIC_WORDS)    # "about Gödel’s ..." -> "Search Wikipedia"
    emb = sl.model.encode([lab, norm(context)])
    sim = float(emb[0] @ emb[1] / ((emb[0] ** 2).sum() ** 0.5 * (emb[1] ** 2).sum() ** 0.5 + 1e-9))
    return score + 0.3 * sim


def assign_fields(goal, fields, taken, history, title=""):
    """Give each named value in the goal to at most one text field, best fits first, so
    "Where from?" gets Zurich, "Where to?" London and "Departure" the date. A field is wanted when
    it received a fitting value it doesn't already hold."""
    values = {v: ctx for v, ctx in goal_values(goal).items() if not any(f" {t} " in f" {norm(v)} " for t in taken)}
    debug = []
    if not fields:
        return debug
    if not values:  # the goal names nothing: only the single most relevant field, e.g. a search box
        c = max(fields, key=lambda c: c["score"])
        if not c["sat"] and SEARCH_LABEL.search(norm(c["a"]["label"])):
            value, conf = field_value(goal, c["a"]["label"].split(" → ")[0], c["a"].get("role", ""), taken)
            c["wanted"], c["value"] = conf >= FIELD_WANTED, value
            debug.append((c["a"]["label"], value, round(conf, 3)))
        return debug
    vals = list(values)
    pairs = []
    for fi, c in enumerate(fields):
        label = c["a"]["label"].split(" → ")[0]
        site = set(norm(title).split())  # a site isn't searched for its own name ("Wikipedia article")
        pairs += [(slot_score(label, v, values[v]) - 0.5 * bool(site & set(norm(v).split())), -fi, fi, vi)
                  for vi, v in enumerate(vals)]
    used_f, used_v = set(), set()
    for sc, _, fi, vi in sorted(pairs, reverse=True):
        if fi in used_f or vi in used_v or sc < SLOT_MIN:
            continue
        used_f.add(fi)
        used_v.add(vi)
        c, value = fields[fi], vals[vi]
        current = c["a"].get("value", "")
        holds = bool(current.strip()) and (in_goal(current, value) or in_goal(value, current))
        c["sat"] = c["sat"] or holds
        c["wanted"] = not holds and norm(value) not in typed_for(history, c["a"]["label"])
        c["value"] = value
        debug.append((c["a"]["label"], value, round(sc, 2)))
    return debug


def field_value(goal, label, role, taken=()):
    """Best value from the goal for a field, and how convincing it is. `taken`: values other controls
    on the page already cover (dropdown options, checkbox labels)."""
    key = (goal, label, tuple(sorted(taken)))
    if key not in _field_cache:
        values = {v: ctx for v, ctx in goal_values(goal).items() if not any(f" {t} " in f" {norm(v)} " for t in taken)}
        if not values:
            values = {p: p for p in goal_phrases(goal)}
        vals = list(values)
        scores = rr.score(TEXT_INSTRUCTION, f'Text field: {role} "{label}"', [values[v] for v in vals])
        i = max(range(len(vals)), key=scores.__getitem__)
        _field_cache[key] = (vals[i], scores[i])
    return _field_cache[key]


# ---------- element context: the text of the card/row/article an element sits in ----------

CONTEXT_JS = """(ids => ids.map(id => {
  const e = window.__jevFast?.nodes.get(id);
  if (!e) return '';
  const box = e.closest('article,li,tr,[role="row"],[role="listitem"],[role="option"],section,form,fieldset,dialog,[role="dialog"]');
  const t = (box && box !== e ? box.innerText : e.parentElement?.innerText || '').replace(/\\s+/g, ' ').trim();
  return t.slice(0, 220);
}))(%s)"""

_observe = Browser.observe


def observe(self, *a, **k):
    page = _observe(self, *a, **k)
    nodes = sorted({x["node"] for x in page["actions"] if "node" in x})
    try:
        texts = self.evaluate(CONTEXT_JS % json.dumps(nodes)) or []
        ctx = dict(zip(nodes, texts))
        for x in page["actions"]:
            c = ctx.get(x.get("node"), "")
            if c and norm(c) != norm(x["label"]):
                x["context"] = c
    except Exception:
        pass  # context is a hint; never fail an observation over it
    return page


Browser.observe = observe


# ---------- candidates ----------

def describe(operation, a):
    label = a["label"].split(" → ")[0]
    role = a.get("role", "")
    state = []
    value = a.get("current_value", a.get("value", ""))
    if value and operation != "SELECT":
        state.append(f'currently "{value[:60]}"')
    for k in ("checked", "selected", "expanded"):
        if k in a:
            state.append(f"{k}={a[k]}")
    tail = f" ({', '.join(state)})" if state else ""
    ctx = " ".join(a.get("context", "").replace(label.strip(), " … ").split())  # what the label doesn't say
    ctx = f' — in: "{ctx[:90]}"' if ctx and norm(ctx) else ""
    if operation == "TYPE_TEXT":
        return f'Type into {role} "{label}"{tail}{ctx}'
    if operation == "SELECT":
        return f'Choose "{a["label"].split(" → ")[-1]}" in dropdown "{label}" (currently "{value}")'
    return f'Click {role} "{label}"{tail}{ctx}'


def history_text(history):
    steps = []
    for h in history[-8:]:
        s = h["action"] + (f' = "{h["text"]}"' if h.get("text") else "")
        if h.get("page_changed") is False:
            s += " (no visible change)"
        steps.append(s)
    return "; ".join(steps) or "nothing yet"


EXPLORE = object()  # marker: wanted only if not opened yet in this run (resolved in choose)


def month_days(text):
    """{(month, day)} named in text, whether written "November 20, 2026" or "Fri, Nov 20"."""
    found = set()
    for m, d in re.findall(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? (\d{1,2})\b", norm(text)):
        found.add((m, int(d)))
    for d, m in re.findall(r"\b(\d{1,2}) (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", norm(text)):
        found.add((m, int(d)))
    return found


def holds_goal_value(value, goal):
    """The field already shows something the goal asks for: same words, or the same date written differently."""
    return bool(value.strip()) and (in_goal(value, goal) or bool(month_days(value) & month_days(goal)))


def input_status(op, a, goal, typed):
    """(is_input, satisfied, wanted) for one candidate. Only lexical/cached checks here."""
    role = a.get("role", "")
    if op == "TYPE_TEXT":
        value = a.get("value", "")
        sat = holds_goal_value(value, goal) or bool(value.strip()) and norm(value) in typed
        return True, sat, None  # wanted decided by field_value() for the few that matter
    if op == "SELECT":
        wanted = in_goal(a["label"].split(" → ")[-1], goal)
        return True, in_goal(a.get("current_value", ""), goal), wanted
    if role in TOGGLES:
        on = str(a.get("checked", "")).lower() == "true"
        return True, on, in_goal(a["label"], goal)
    if role == "combobox" and op == "CLICK" and not a["label"].startswith("Open "):
        parts = [a.get("value", "")] + re.split(r"[.,:]\s+", a["label"])
        sat = any(in_goal(x, goal) for x in parts if x.strip())
        # Its options are hidden until opened: one whose value isn't in the goal gets opened once.
        return True, sat, not sat and len(norm(a.get("value", "")).split()) in (1, 2, 3) and EXPLORE
    if op == "CLICK" and role in ("combobox", "textbox", "searchbox") and holds_goal_value(a.get("value", ""), goal):
        # Re-opening a field that already holds the goal's value ("Open Where from?" = Zurich) isn't progress.
        return True, True, False
    return False, False, False


def _combo_wanted(a, goal):
    # A non-editable combobox (e.g. a ticket-type picker) is wanted when one of the goal's words
    # names a different value than the current one; we can only see its label here.
    return any(in_goal(w, goal) for w in re.split(r"[.,]", a["label"]) if w.strip())


# ---------- the decision ----------

def choose(state, goal, history):
    """Reuse the previous decision when the page only changed cosmetically (prices loading,
    animations) and the chosen element is still there; otherwise decide afresh."""
    prev = _reuse.get(id(history))
    reuses = _reuse_count.get(id(history), (None, 0))
    if prev and prev[0] == len(history) and not (reuses[0] == len(history) and reuses[1] >= 3):
        kind, label = prev[1]
        same = [x for x in state["actions"] if (x["kind"], x["label"]) == (kind, label)]
        if same and prev[2]["operation"] not in ("DONE", "BLOCKED", "WAIT"):
            answer = dict(prev[2], choice=same[0]["id"], latency_ms=0)
            answer["probabilities"] = {same[0]["id"]: answer["confidence"]}
            answer["raw_answers"] = dict(answer["raw_answers"], reason=answer["raw_answers"]["reason"] + " (reused)")
            n = reuses[1] + 1 if reuses[0] == len(history) else 1
            _reuse_count[id(history)] = (len(history), n)
            return answer
    answer = decide(state, goal, history)
    chosen = next((x for x in state["actions"] if x["id"] == answer["choice"]), None)
    if chosen:
        _reuse[id(history)] = (len(history), (chosen["kind"], chosen["label"]), answer)
    return answer


def decide(state, goal, history):
    started = time.perf_counter()
    _timing.clear()
    _debug.clear()
    elements, targets, controls = action_space(state["actions"])
    order = {e["index"]: i for i, e in enumerate(elements)}
    cands = []  # dicts: op, index, a, doc
    for op, group in targets.items():
        for index, a in group.items():
            cands.append({"op": op, "index": index, "a": a, "doc": describe(op, a)})

    typed = {norm(h["text"]) for h in history if h.get("text")}
    recent = [(h["operation"], h["action"]) for h in history[-4:]]
    stuck = {(h["operation"], h["action"]) for h in history if h.get("page_changed") is False}
    query = f"Goal: {goal}\nCurrent page: {state['title']}\nSteps taken: {history_text(history)}"

    # Cheap shortlist on busy pages; inputs always stay in.
    if len(cands) > SHORTLIST_OVER:
        keep = set(sl.top(query, [c["doc"] for c in cands], SHORTLIST_K))
        typed_words = set(norm(history[-1].get("text") or "").split()) if history else set()
        if history and history[-1]["kind"] == "wait" and len(history) > 1:
            typed_words = set(norm(history[-2].get("text") or "").split())
        cands = [c for i, c in enumerate(cands) if i in keep or c["op"] in ("TYPE_TEXT", "SELECT")
                 or c["a"].get("role") in TOGGLES or c["a"].get("role") in ("combobox", "option")
                 or typed_words & set(norm(c["a"]["label"]).split())]

    scores = rr.score(ACTION_INSTRUCTION, query, [c["doc"] for c in cands]) if cands else []
    for c, s in zip(cands, scores):
        c["score"] = c["raw"] = s  # raw: the reranker's own score, before any rule touches it
        c["is_input"], c["sat"], c["wanted"] = input_status(c["op"], c["a"], goal, typed)
        # Repeats: the same operation on the same label in the last few steps is demoted.
        if (c["op"], c["a"]["label"]) in recent:
            c["score"] *= 0.01 if (c["op"], c["a"]["label"]) in stuck else 0.1
        if c["is_input"] and c["sat"]:
            c["score"] *= 0.05

    explored = _explored.setdefault(id(history), set())
    for c in cands:
        if c["wanted"] is EXPLORE:
            c["wanted"] = combo_key(c["a"]) not in explored

    # Among visible options (dropdown items, listbox entries), one named in the goal wins.
    opts = [c for c in cands if c["a"].get("role") in OPTION_ROLES and c["op"] == "CLICK"]
    if any(in_goal(c["a"]["label"], goal) for c in opts):
        for c in opts:
            c["score"] = c["score"] + 0.5 if in_goal(c["a"]["label"], goal) else c["score"] * 0.1

    # Text fields: wanted if the goal contains a convincing value for them (top few by relevance).
    taken = {norm(c["a"]["label"].split(" → ")[-1]) for c in cands
             if (c["op"] == "SELECT" or c["a"].get("role") in TOGGLES) and in_goal(c["a"]["label"].split(" → ")[-1], goal)}
    fields_debug = assign_fields(goal, [c for c in cands if c["op"] == "TYPE_TEXT"], taken, history, state["title"])

    # Named things in the goal not acted on yet ("It's Only the Himalayas") beat ones already done
    # ("Travel") and generic words ("Books" for "the book").
    done_text = norm(" ".join(h["action"] + " " + (h.get("text") or "") for h in history))
    for c in cands:
        if c["op"] != "CLICK" or c["is_input"]:
            continue
        lab = set(norm(c["a"]["label"]).split())
        for v in goal_values(goal):
            toks = [t for t in norm(v).split() if t not in STOP]
            if toks and sum(t in lab for t in toks) >= max(1, len(toks) / 2) \
                    and not all(t in done_text.split() for t in toks):
                c["score"] += 0.5
                break

    last = history[-1] if history else None
    pending = sorted((c for c in cands if c["is_input"] and not c["sat"] and c["wanted"]),
                     key=lambda c: order.get(c["index"].split(":")[0], 0))
    # Remember requested inputs across pages: one seen earlier and never acted on still blocks Done.
    memory = _pending_memory.setdefault(id(history), {})
    for c in pending:
        memory.setdefault(input_key(c), c["doc"][:80])
    for c in cands:
        if c["is_input"] and c["sat"]:
            memory.pop(input_key(c), None)
    for h in history:
        memory.pop(h["action"].split(" → ")[0], None)
    pick, reason = None, "goal"

    # The site answered with an error page ("Oops, something went wrong"): click its own retry button
    # (or redo the click that led here), pausing longer before each try like a person would, up to
    # MAX_ERROR_RETRIES times; then give up instead of wandering off and redoing the form.
    if last and ERROR_RE.search(state.get("text", "")[:3000]):
        tries = _error_retries.setdefault(id(history), [])
        retry = [c for c in cands if c["op"] == "CLICK" and RETRY_RE.search(c["a"]["label"])]
        again = [c for c in cands if c["op"] == "CLICK" and c["a"]["label"] == last["action"]
                 and last["operation"] == "CLICK"]
        if len(tries) >= MAX_ERROR_RETRIES:
            pick, reason = "BLOCKED", f"page still shows an error after {len(tries)} retries"
        elif retry or again:
            time.sleep(2 * (len(tries) + 1))
            tries.append(len(history))
            pick = (retry or again)[0]
            reason = f"page shows an error: {'click its retry button' if retry else 'retry the last click'} (try {len(tries)})"
        elif "WAIT" in controls:
            time.sleep(2)
            tries.append(len(history))
            pick, reason = "WAIT", "page shows an error: wait"

    # 0. suggestions after typing (possibly after one wait) / opening a combobox
    typed_step = history[-2] if last and last["kind"] == "wait" and len(history) > 1 and history[-2]["kind"] == "fill" \
        else last
    if not pick and typed_step and (typed_step["kind"] == "fill" or typed_step["operation"] == "CLICK" and typed_step.get("page_changed")):
        last = typed_step
        text = norm(last.get("text") or "")
        first = text.split()[0] if text else ""
        options = [c for c in cands if c["op"] == "CLICK" and not c["is_input"]
                   and c["a"].get("role") not in ("textbox", "searchbox", "combobox") and c["a"].get("role") not in TOGGLES
                   and (c["a"].get("role") in OPTION_ROLES or first and f" {first} " in f' {norm(c["a"]["label"])} ')
                   and c["a"]["label"] != last["action"]]
        if options:
            # Ranked by what was typed first, the reranker only breaks ties: it prefers generic entries
            # like "Search for pages containing ..." over the actual match.
            want = text.split()

            def fit(c):
                got = norm(c["a"]["label"]).split()
                overlap = sum(w in got for w in want) / len(want) if want else 0
                extra = len([w for w in got if w not in want])  # fewest added words = closest to what was typed
                return (norm(c["a"]["label"]).startswith(first), round(overlap, 2), -extra, c["score"])
            pick, reason = max(options, key=fit), "suggestion after typing"
            if last["kind"] != "fill":
                reason = "option from the dropdown just opened"
                if not any(in_goal(c["a"]["label"], goal) for c in options):
                    keep = [c for c in options if norm(c["a"]["label"]) in norm(last["action"])]
                    pick = keep[0] if keep else None  # nothing asked for here: keep what was selected
            _debug["options"] = [(fit(c)[:2], c["a"]["label"][:60]) for c in sorted(options, key=fit, reverse=True)[:5]]

    last = history[-1] if history else None

    # 2. submit after typing
    typed_into = next((x for x in state["actions"] if x["label"] == last["action"]), None) if last else None
    # Wait for suggestions after typing into a search/combobox, or into a field that got renamed
    # while they load (Google turns "Where from?" into "Where else?").
    if not pick and last and last["kind"] == "fill" and "WAIT" in controls \
            and (typed_into is None or typed_into.get("role") in ("searchbox", "combobox")):
        pick, reason = "WAIT", "let suggestions load after typing"
    elif not pick and last and last["kind"] == "fill" or (
            not pick and last and last["kind"] == "wait" and len(history) > 1 and history[-2]["kind"] == "fill"):
        last = history[-2] if last["kind"] == "wait" else last
        buttons = [c for c in cands if c["op"] == "CLICK" and c["a"].get("role") in ("button", "link")
                   and not c["is_input"]]
        field_ctx = norm(next((x.get("context", "") for x in state["actions"] if x["label"] == last["action"]), ""))
        same_form = [c for c in buttons if norm(c["a"]["label"]) and f" {norm(c['a']['label'])} " in f" {field_ctx} "]
        if same_form:
            pick, reason = max(same_form, key=lambda c: c["score"]), "submit: button in the same form"
        elif buttons:
            sub = rr.score(SUBMIT_INSTRUCTION, f'Field: "{last["action"]}" = "{last.get("text", "")}"',
                           [c["doc"] for c in buttons])
            i = max(range(len(buttons)), key=sub.__getitem__)
            if sub[i] > 0.3:
                pick, reason = buttons[i], f"submit after typing (p={sub[i]:.2f})"

    # 1. requested inputs, top to bottom
    if not pick and pending:
        pick, reason = pending[0], "requested input not set yet"
        if pick["a"].get("role") == "combobox" and pick["op"] == "CLICK":
            explored.add(combo_key(pick["a"]))
            reason = "open dropdown to see its options"

    # Done: its own question about the page, only when no requested input is pending.
    p_done = done_probability(state, goal, _last_pick.get(id(history))) if history and not pending and not memory else 0.0

    # 3. most relevant element
    best = max(cands, key=lambda c: c["score"]) if cands else None
    best_score = best["score"] if best else 0.0
    if not pick and p_done >= DONE_THRESHOLD:
        pick, reason = "DONE", f"page shows the goal achieved (p={p_done:.2f})"
    elif not pick and "WAIT" in controls and [h["kind"] for h in history[-2:]].count("wait") < 2 and (
            (_debug.get("done_parts") or {}).get("names", [0, False])[1] or p_done >= 0.3):
        # The page is the right one (title/URL name the goal) but the evidence isn't there yet: let it load.
        pick, reason = "WAIT", f"right page, results still loading (p={p_done:.2f})"
    if not pick and best and best_score < ELEMENT_FLOOR:
        obstacle = rr.score(OBSTACLE_INSTRUCTION, OBSTACLE_QUERY, [c["doc"] for c in cands])
        i = max(range(len(cands)), key=obstacle.__getitem__)
        if obstacle[i] > 0.5:
            pick, reason = cands[i], f"obstacle in the way (p={obstacle[i]:.2f})"
    if not pick and (not best or best_score < ELEMENT_FLOOR):
        recent_ids = [h["choice"] for h in history[-3:]]
        if "SCROLL_DOWN" in controls and recent_ids.count("scroll_down") < 3:
            pick, reason = "SCROLL_DOWN", "nothing relevant visible"
        elif "WAIT" in controls and recent_ids[-1:] != ["wait"]:
            pick, reason = "WAIT", "nothing relevant visible"
        elif not best:
            pick, reason = "BLOCKED", "nothing to act on"
    if not pick:
        pick = best
    if isinstance(pick, dict) and reason in ("goal", "suggestion after typing"):
        _last_pick[id(history)] = (pick["a"]["label"], pick["score"], state["title"])

    reranker_top = max(cands, key=lambda c: c["raw"]) if cands else None
    # ---------- shape the answer jev-ultrafast expects ----------
    if isinstance(pick, str):
        operation, target = pick, None
        choice = controls[pick]["id"] if pick in controls else pick
    else:
        operation, target, choice = pick["op"], pick["index"], pick["a"]["id"]
        if pick.get("value"):
            _next_text[pick["a"]["label"]] = pick["value"]
    raw = [c["score"] for c in cands] or [0.0]
    m = max(raw)
    exp = [math.exp((s - m) * 20) for s in raw]
    tot = sum(exp)
    probs = {c["a"]["id"]: e / tot for c, e in zip(cands, exp)}
    if choice not in probs:
        probs = {choice: 1.0}
    op_probs = {}
    for c in cands:
        op_probs[c["op"]] = op_probs.get(c["op"], 0) + probs.get(c["a"]["id"], 0)
    op_probs[operation] = max(op_probs.get(operation, 0), probs.get(choice, 0))
    target_probs = {c["index"]: probs[c["a"]["id"]] for c in cands if c["op"] == operation}
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": probs.get(choice, 1.0),
        "probabilities": {c["a"]["id"]: probs[c["a"]["id"]] for c in cands if c["op"] == operation} or {choice: 1.0},
        "operation_probabilities": op_probs,
        "target_probabilities": target_probs,
        "target_confidence": probs.get(choice) if target else None,
        "raw_answers": {
            "reason": reason, "p_done": round(p_done, 4),
            "pending": [c["doc"][:80] for c in pending] + [f"(earlier) {v}" for v in memory.values()],
            "fields": fields_debug, **_debug,
            "timing": list(_timing) + [("total before shaping", 0, round((time.perf_counter() - started) * 1000))],
            "ranked": sorted(([round(c["score"], 4), c["doc"][:120]] for c in cands), reverse=True)[:10],
            # What the reranker alone would have done here, to show whether a rule changed anything.
            "raw_ranked": sorted(([round(c["raw"], 4), c["doc"][:140]] for c in cands), reverse=True)[:12],
            "picked": pick["doc"][:140] if isinstance(pick, dict) else pick,
            "reranker_top": reranker_top["a"]["label"][:80] if reranker_top else None,
            "reranker_top_op": reranker_top["op"] if reranker_top else None,
            "reranker_agreed": bool(reranker_top) and isinstance(pick, dict) and pick is reranker_top,
        },
        "model": f"{MODEL.split('/')[-1]} (local)",
        "usage": {"candidates": len(cands)},
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": {"query": query, "candidates": len(cands)},
    }


_pending_memory = {}  # id(run history) -> {input label: description}
_explored = {}        # id(run history) -> dropdowns already opened to see their options
_reuse = {}           # id(run history) -> (history length, (kind, label), answer) of the last decision
_reuse_count = {}     # id(run history) -> (history length, consecutive reuses)


def combo_key(a):
    return norm(re.split(r"[.,:]\s+", a["label"])[0])


def input_key(c):
    return c["a"]["label"].split(" → ")[0]


_last_pick = {}  # id(run history) -> (label, relevance) of the last element chosen on goal relevance


def done_probability(state, goal, last_pick=None):
    """Has the page reached the goal? Judged on the page's identity (title + URL) against the goal's
    last step, not on body text (menus and links mention the goal everywhere). A "Stop when ..."
    sentence is checked against the visible text instead."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", goal.strip()) if x.strip()]
    sentences = [x for x in sentences if not re.match(r"(do not|don't|never)\b", x, re.I)]
    stop = [x for x in sentences if re.match(r"stop (when|once|after)\b", x, re.I)]
    main = [x for x in sentences if x not in stop] or sentences
    final = re.split(r",? (?:and then|then|and) ", main[-1])[-1]
    url = state["url"].split("?")[0].split("#")[0] + ("#" + state["url"].split("#")[1] if "#" in state["url"] else "")
    ident = f'Page title: {state["title"]}\nURL: {url}'  # no query string: it can be hundreds of tokens
    p = rr.score(DONE_INSTRUCTION, f"Goal: {final}", [ident], max_doc_tokens=96)[0]
    _debug["done_parts"] = {"final": final, "ident": round(p, 3)}
    names = sorted(goal_values(final), key=final.find)  # in the order the goal says them
    if names:  # the last step names something: it should be on the page we're on
        here = norm(state["title"] + " " + url.replace("_", " ").replace("-", " "))
        target = norm(names[-1]).split()  # the last thing named is what the goal ends on
        hit = sum(t in here.split() for t in target) >= max(1, len(target) / 2)
        proper = names[-1][:1].isupper() or bool(DATE_RE.search(names[-1]))  # a name must appear literally;
        p = max(p, 0.9) if hit else (p * 0.3 if proper else p)               # a lowercase topic need not
        _debug["done_parts"]["names"] = (names, hit, round(p, 3))
    # "Open X": the relevant link we just clicked is now the page's title.
    if last_pick and re.match(r"(open|go to|visit|view|show|read|navigate)\b", final.strip(), re.I):
        label, relevance, title_before = last_pick
        words = [w for w in norm(label).split() if w not in STOP]
        if names:  # the click must be on the last thing the goal names ("Olio", not "Poetry")
            target = norm(names[-1]).split()
            object_ok = any(t in norm(label).split() for t in target if t not in STOP)
        else:  # nothing named: the clicked link must still use the goal's words ("jobs" for "the jobs page")
            object_ok = any(w in norm(final).split() for w in words)
        new_in_title = all(w in norm(state["title"]).split() for w in words) and \
            not all(w in norm(title_before).split() for w in words)  # a home/logo link was already there
        if relevance >= 0.3 and object_ok and words and new_in_title:
            p = max(p, 0.9)
    if stop:  # check the condition against the visible text in chunks: results are rarely at the top
        body = state.get("text", "")[:4800]
        chunks = [f"{ident}\nText: {body[i:i + 800]}" for i in range(0, max(len(body), 1), 800)]
        # Results are often buttons/links ("Select flight ..."), so their labels are evidence too.
        labels = [x["label"] for x in state["actions"] if x.get("kind") == "click"]
        for i in range(0, min(len(labels), 60), 15):
            chunks.append(f"{ident}\nPage items: " + " | ".join(lab[:120] for lab in labels[i:i + 15]))
        # Evidence: lines and items that mention what the goal names (Zurich, London, the date).
        tokens = {t for n in goal_values(goal) for t in norm(n).split() if len(t) > 2}
        lines = [x for x in state.get("text", "").splitlines() + labels if tokens & set(norm(x).split())]
        if lines:
            chunks.append(f"{ident}\nMatching items: " + " | ".join(x[:160] for x in lines[:12]))
        if re.search(r"visible|shown|appear|listed|displayed|loaded", stop[0], re.I):
            # "results are visible": the reranker can't tell a results page from a filled search form,
            # so count result-like items: non-input elements naming at least two of the goal's values.
            values = [[t for t in norm(v).split() if not t.isdigit() or len(t) <= 2] for v in goal_values(goal)]
            items = [x["label"] for x in state["actions"] if x.get("kind") == "click"
                     and x.get("role") not in ("textbox", "searchbox", "combobox", "option", "gridcell")]
            hits = [lab for lab in items if sum(
                sum(t in norm(lab).split() for t in v) >= max(1, len(v) / 2) for v in values if v) >= 2]
            _debug["done_parts"]["result_items"] = [len(hits)] + [h[:100] for h in hits[:3]]
            p = min(p, 0.9 if len(hits) >= 3 else 0.1)
        else:
            stop_scores = rr.score(DONE_INSTRUCTION, f"Goal: {stop[0]}", chunks, max_doc_tokens=320)
            _debug["done_parts"]["stop"] = [round(x, 3) for x in stop_scores]
            p = min(p, max(stop_scores))
    return p


def typed_for(history, label):
    return {norm(h["text"]) for h in history if h.get("text") and h["action"] == label}


_next_text = {}  # field label -> value already chosen while deciding


def field_text(context):
    started = time.perf_counter()
    goal, field = context["goal"], context["field"]
    label = (field.get("label") or "").split(" → ")[0]
    value = _next_text.pop(field.get("label") or "", None) or field_value(goal, label, field.get("role", ""))[0]
    return value, {"model": "rules: phrase copied from the goal",
                   "latency_ms": round((time.perf_counter() - started) * 1000), "usage": {}}
