"""jev-ultrafast with a local reranker instead of TypeSafe's Jev (and no text LLM).

Plugs two functions into jev-ultrafast's agent loop, leaving its repo untouched:
  choose()      every (operation, element) pair is a candidate; the reranker scores each one
                against the goal + steps so far and the best is executed.
  field_text()  the value to type is a phrase copied from the goal, matched to the field by simple rules.

    uv run local_jev.py           # simple UI (agent.html) on http://127.0.0.1:8766, their agent loop underneath
"""
import os
import sys
import time
from pathlib import Path

def agent_chrome_ws(port=9333):
    """Browser websocket of the separate agent Chrome (own profile, started by start_chrome.sh)."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as r:
            return json.load(r)["webSocketDebuggerUrl"]
    except OSError:
        sys.exit(f"No agent Chrome on port {port}. Run ./start_chrome.sh first.")


os.environ.setdefault("BU_CDP_WS", agent_chrome_ws())
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")
sys.path.insert(0, str(Path(__file__).parent / "jev-ultrafast"))

import jev_ultrafast.agent as jev_agent  # noqa: E402
from jev_ultrafast import demo  # noqa: E402
# Three decision modes share one loaded reranker; each run picks one (UI menu, or AGENT=... for the benchmark):
#   rules   reranker + hand-written generic rules that can override it (decide.py)
#   assist  code only prepares the options, the reranker always makes the choice (assist.py)
#   pure    the reranker alone, with only the help jev-ultrafast gives Jev (pure.py)
#   pure_done  the reranker alone for every move, plus the rules version's done check (pure_done.py)
import assist  # noqa: E402
import decide  # noqa: E402
import pure  # noqa: E402
import pure_done  # noqa: E402

MODES = {"rules": decide, "assist": assist, "pure": pure, "pure_done": pure_done}
MODE = {"current": os.environ.get("AGENT", "rules") if os.environ.get("AGENT") in MODES else "rules"}


def choose(state, goal, history):
    return MODES[MODE["current"]].choose(state, goal, history)


def field_text(context):
    return MODES[MODE["current"]].field_text(context)


jev_agent.choose = choose
jev_agent.field_text = field_text

# Speed accounting: time spent in the browser, kept apart from model time (decisions' latency_ms).
#   open     first navigation to the start URL
#   act      executing each action (mouse/keyboard, incl. the human-like pointer movement)
#   observe  waiting for the page to settle, reading its elements, screenshot
import jev_ultrafast.browser as _jb  # noqa: E402


def _timed(name, fn):
    def wrapper(self, *a, **k):
        t = time.perf_counter()
        try:
            return fn(self, *a, **k)
        finally:
            speed = self.__dict__.setdefault("_speed", {"open": 0.0, "act": [], "observe": []})
            ms = (time.perf_counter() - t) * 1000
            if name == "open":
                speed["open"] += ms
            else:
                speed[name].append(ms)
    return wrapper


_jb.Browser.__init__ = _timed("open", _jb.Browser.__init__)
_jb.Browser.act = _timed("act", _jb.Browser.act)
_jb.Browser.observe = _timed("observe", _jb.Browser.observe)


def speed_summary():
    """Where a run's time went, plus how many options the model scored."""
    agent = demo.AGENT
    if not agent:
        return None
    speed = getattr(agent.state.get("browser"), "_speed", None) or agent.__dict__.get("_speed_final")
    if not speed:
        return None
    decisions = agent.state["decisions"]
    counts = [d.get("usage", {}).get("candidates", 0) for d in decisions]
    model = sum(d.get("latency_ms", 0) for d in decisions) + sum(c.get("latency_ms", 0) for c in agent.state["text_calls"])
    reused = sum(1 for d in decisions if d.get("latency_ms") == 0)
    return {
        "model_ms": round(model), "decisions": len(decisions), "reused": reused,
        "text_calls": len(agent.state["text_calls"]),
        "candidates_total": sum(counts), "candidates_avg": round(sum(counts) / len(counts), 1) if counts else 0,
        "candidates_max": max(counts, default=0),
        "open_ms": round(speed["open"]), "act_ms": round(sum(speed["act"])), "acts": len(speed["act"]),
        "observe_ms": round(sum(speed["observe"])), "observes": len(speed["observe"]),
    }

# Closing a run's tab that was already closed by hand made every later reset fail with
# "No target with given id found", so a tab that is already gone counts as closed.
import jev_ultrafast.browser as jev_browser  # noqa: E402
_close_tab = jev_browser.Browser.close


def close_tab(self):
    try:
        _close_tab(self)
    except Exception as ex:
        if "No target with given id" not in str(ex):
            raise
        self.target = None


jev_browser.Browser.close = close_tab

# Their clicks are a bare press+release at the element's exact centre, with no mouse movement and no
# hold. Google Flights' "Reload" ignored those every time but worked for a person's click, so clicks
# now arrive like a person's: the pointer moves over in a few steps, rests, and the press lasts a moment.
import random  # noqa: E402

_cdp = jev_browser.cdp
_pointer = {}  # session -> last (x, y)


def human_cdp(method, session_id=None, **params):
    if method == "Input.dispatchMouseEvent" and params.get("type") == "mousePressed":
        x, y = params["x"], params["y"]
        x0, y0 = _pointer.get(session_id, (x - random.uniform(80, 200), y + random.uniform(40, 120)))
        for i in range(1, 6):
            t = i / 5
            _cdp("Input.dispatchMouseEvent", session_id=session_id, type="mouseMoved",
                 x=x0 + (x - x0) * t + random.uniform(-2, 2) * (1 - t), y=y0 + (y - y0) * t + random.uniform(-2, 2) * (1 - t))
            time.sleep(random.uniform(0.01, 0.03))
        time.sleep(random.uniform(0.05, 0.12))
        _pointer[session_id] = (x, y)
        result = _cdp(method, session_id=session_id, **params)
        time.sleep(random.uniform(0.06, 0.12))
        return result
    return _cdp(method, session_id=session_id, **params)


jev_browser.cdp = human_cdp
TEXT_MODELS = {"rules": "rules, copied from the goal", "assist": "reranker picks value and field together",
               "pure": "reranker picks a phrase of the goal", "pure_done": "reranker picks a phrase of the goal"}
os.environ["TEXT_MODEL"] = TEXT_MODELS[MODE["current"]]

# Extra sites besides their three scenarios, added by wrapping demo.py instead of editing it.
SITES = {
    "wikipedia": "https://en.wikipedia.org/wiki/Main_Page",
    "books": "https://books.toscrape.com/",
    "hn": "https://news.ycombinator.com/",
    "custom": "",
}
_command = demo.command


def command(name, body):
    scenario = body.get("scenario")
    if name == "reset":
        mode = body.get("mode", MODE["current"])
        if mode not in MODES:
            raise ValueError("Unknown mode")
        MODE["current"] = mode
        os.environ["TEXT_MODEL"] = TEXT_MODELS[mode]
    if name != "reset" or scenario not in SITES:
        state = _command(name, body)
        state["mode"] = MODE["current"]
        state["speed"] = speed_summary()
        return state
    url = body.get("url", "").strip() if scenario == "custom" else SITES[scenario]
    if not url.startswith(("http://", "https://")):
        raise ValueError("Enter a full URL starting with https://")
    goal = body.get("goal", "").strip()
    if not goal or len(goal) > 2000:
        raise ValueError("Enter 1–2,000 characters")
    demo.close_browser()
    demo.AGENT = jev_agent.Agent(url, goal, screenshots=True)
    demo.AGENT.state["scenario"] = scenario
    return {**demo.response_state(), "mode": MODE["current"], "speed": speed_summary()}


demo.command = command
_read_text = Path.read_text
PAGE = Path(__file__).parent / "agent.html"


def read_text(self, *a, **k):
    """Serve our own simpler page (agent.html) in place of their inspector UI."""
    if self.name == "index.html" and self.parent.name == "static":
        return _read_text(PAGE, *a, **k)
    return _read_text(self, *a, **k)


Path.read_text = read_text

if __name__ == "__main__":
    os.chdir(Path(__file__).parent / "jev-ultrafast")
    demo.main()
