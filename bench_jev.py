"""Benchmark the local reranker agent on jev-ultrafast's tasks plus a few real sites.

One process: serves their fixture page, drives the agent Chrome, checks outcomes
independently (never trusting the agent's own DONE), saves traces.

    ./start_chrome.sh            # once
    uv run bench_jev.py          # all tasks
    uv run bench_jev.py travel books
"""
import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import local_jev  # noqa: F401  (patches choose/field_text, loads the reranker)
from jev_ultrafast import demo
from jev_ultrafast.agent import Agent

sys.path.insert(0, str(Path(__file__).parent / "jev-ultrafast" / "examples"))
from flights import URL as FLIGHTS_URL  # noqa: E402

# Their task asks for September 20, 2026, which is now in the past (Google won't offer it), so the
# same task moves to November 20, 2026 and is checked the same way their flights.py verify() does.
FLIGHTS_GOAL = ("Find one-way flights from Zurich to London on November 20, 2026, for one adult in economy. "
                "Stop when matching flight options are visible. Do not select or book a flight.")


def verify_flights(page):
    from urllib.parse import urlparse
    values = {a["label"].strip(): a.get("value") for a in page["actions"]}
    flights = [a["label"] for a in page["actions"] if "Select flight" in a["label"]]
    return {"passed": urlparse(page["url"]).path == "/travel/flights/search"
            and values.get("Change ticket type. One way") == "One way"
            and values.get("Where from?") == "Zürich" and values.get("Where to?") == "London"
            and values.get("Departure") == "Fri, Nov 20"
            and bool(flights) and all("Friday, November 20" in f for f in flights)}

FIXTURE = f"http://127.0.0.1:{demo.PORT}/fixture.html"


def body(agent):
    return agent.browser.evaluate("document.body.innerText") or ""


TASKS = {
    "travel": (f"{FIXTURE}?scenario=travel", "Find a Design stay in Lisbon with Free cancellation and open Casa Flora.",
               lambda a, s: s["page"]["url"].endswith("#casa-flora")
               and "Your filters: Design · Free cancellation enabled · Destination Lisbon" in body(a)),
    "research": (f"{FIXTURE}?scenario=research", "Open the article about using finite choices to control browser agents.",
                 lambda a, s: s["page"]["title"].startswith("A browser is a choice")),
    "wikipedia": ("https://en.wikipedia.org/wiki/Main_Page",
                  "Find and open the Wikipedia article about Gödel’s incompleteness theorems.",
                  lambda a, s: unquote(s["page"]["url"]).endswith("/wiki/Gödel's_incompleteness_theorems")),
    "books": ("https://books.toscrape.com/", "Open the Poetry category and then open the book Olio.",
              lambda a, s: "/olio_" in s["page"]["url"]),
    "hn": ("https://news.ycombinator.com/", "Open the Ask HN page.",
           lambda a, s: s["page"]["url"].rstrip("/").endswith("/ask")),
    "flights": (FLIGHTS_URL, FLIGHTS_GOAL, lambda a, s: verify_flights(s["page"])["passed"]),
    # Written after the six main tasks. Not held out: running them led to a fix.
    "x-books": ("https://books.toscrape.com/", "Open the Travel category and then open the book It's Only the Himalayas.",
                lambda a, s: "/its-only-the-himalayas_" in s["page"]["url"]),
    "x-wiki": ("https://en.wikipedia.org/wiki/Main_Page", "Find and open the Wikipedia article about the Eiffel Tower.",
               lambda a, s: unquote(s["page"]["url"]).endswith("/wiki/Eiffel_Tower")),
    "x-hn": ("https://news.ycombinator.com/", "Open the jobs page.",
             lambda a, s: s["page"]["url"].rstrip("/").endswith("/jobs")),
    "x-quotes": ("https://quotes.toscrape.com/", "Open the quotes tagged love.",
                 lambda a, s: "/tag/love" in s["page"]["url"]),
    # Written after the x- tasks had led to a fix. Not held out either: y-wiki later exposed a
    # regression (a suggestion-matching change) that was fixed.
    "y-travel": (f"{FIXTURE}?scenario=travel", "Find a Nature stay in Lisbon and open Serra Lodge.",
                 lambda a, s: s["page"]["url"].endswith("#serra-lodge")
                 and "Your filters: Nature · Free cancellation off · Destination Lisbon" in body(a)),
    "y-books": ("https://books.toscrape.com/", "Open the Mystery category and then open the book Sharp Objects.",
                lambda a, s: "/sharp-objects_" in s["page"]["url"]),
    "y-wiki": ("https://en.wikipedia.org/wiki/Main_Page", "Find and open the Wikipedia article about photosynthesis.",
               lambda a, s: unquote(s["page"]["url"]).endswith("/wiki/Photosynthesis")),
    "y-quotes": ("https://quotes.toscrape.com/", "Open the author page of Albert Einstein.",
                 lambda a, s: "/author/Albert-Einstein" in s["page"]["url"]),
}
LATER = [k for k in TASKS if k.startswith(("x-", "y-"))]  # written later; see the comments above


def run(name, max_actions=int(os.environ.get("MAX_ACTIONS", 30))):
    url, goal, check = TASKS[name]
    out = Path(__file__).parent / "traces" / name
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    agent = Agent(url, goal, record_dir=out / "frames" if os.environ.get("FRAMES") else None)
    error = None
    try:
        for state in agent.run():
            if len(state["history"]) >= max_actions:
                error = f"stopped at {max_actions} actions"
                break
    except Exception as ex:  # report, don't crash the benchmark
        error = f"{type(ex).__name__}: {ex}"
    state = agent.snapshot()
    try:
        ok = bool(check(agent, state))
    except Exception as ex:
        ok, error = False, error or f"check failed: {ex}"
    secs = time.time() - t
    (out / "state.json").write_text(json.dumps(state, indent=2, default=str))
    def decided_for(h):  # decisions on a page that changed meanwhile are redone, so match by time
        return [d for d in state["decisions"] if d["elapsed_ms"] <= h["executed_ms"]][-1]
    steps = "\n      ".join(
        f'{h["kind"]}:{h["action"][:40]}' + (f'="{h["text"]}"' if h.get("text") else "")
        + f'   [{decided_for(h)["raw_answers"].get("reason", "")}]'
        for h in state["history"])
    final = state["decisions"][-1]["raw_answers"] if state["decisions"] else {}
    steps += f'\n      last decision: {final.get("reason")} p_done={final.get("p_done")} pending={final.get("pending")} fields={final.get("fields")}'
    agent.close()
    model_ms = sum(d["latency_ms"] for d in state["decisions"])
    print(f'{"PASS" if ok else "FAIL"} {name:<10} {len(state["history"]):2d} actions {secs:5.1f}s '
          f'(model {model_ms / 1000:.1f}s) status={state["status"]}{" err=" + error if error else ""}\n'
          f'      {steps}', flush=True)
    return ok


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", demo.PORT), demo.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    names = sys.argv[1:] or [k for k in TASKS if k not in LATER]
    if names == ["later"]:
        names = LATER
    elif names == ["all"]:
        names = list(TASKS)
    results = {n: run(n) for n in names}
    print(f"\n{sum(results.values())}/{len(results)} passed")
    server.shutdown()
