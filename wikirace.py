"""Wikirace with a local reranker as the decision model (no training).

Each hop: fetch every article linked from the current page (with its short
description), score each one against the target, follow the best unvisited link.

    uv run wikirace.py                       # built-in races
    uv run wikirace.py "Pizza" "Quantum mechanics"
"""
import sys
import time

import requests

from reranker import Reranker
from shortlist import Shortlist

API = "https://en.wikipedia.org/w/api.php"
S = requests.Session()
S.headers["User-Agent"] = "RS-1-wikirace/0.1 (https://github.com/mattiaongit/rs-1-reranker-decision-model)"

INSTRUCTION = (
    "Given a target Wikipedia article, judge whether the Document is in the same field "
    "of knowledge and would mention or link to the target."
)

# Citation/identifier links present on almost every article; never useful hops.
BOILERPLATE = {
    "ISBN", "ISSN", "Digital object identifier", "Semantic Scholar", "Google Books", "Wayback Machine",
    "PubMed", "PubMed Central", "OCLC", "JSTOR", "Bibcode", "ArXiv", "S2CID", "Library of Congress Control Number",
    "Integrated Authority File", "Virtual International Authority File", "Wikidata", "Doi (identifier)",
}

RACES = [
    ("Pizza", "Quantum mechanics"),
    ("Banana", "Roman Empire"),
    ("Taylor Swift", "Photosynthesis"),
    ("Lego", "Mount Everest"),
    ("Chess", "Volcano"),
    ("Coffee", "Apollo 11"),
    ("Penguin", "French Revolution"),
    ("Bicycle", "William Shakespeare"),
]


def api(**params):
    params |= {"action": "query", "format": "json", "formatversion": 2}
    while True:
        r = S.get(API, params=params, timeout=30).json()
        yield r
        if "continue" not in r:
            return
        params |= r["continue"]


def resolve(title):
    page = next(api(titles=title, redirects=1, prop="description"))["query"]["pages"][0]
    return page["title"], page.get("description", "")


def links(title):
    """{title: description} for every main-namespace article linked from `title`."""
    out = {}
    for r in api(generator="links", titles=title, gplnamespace=0, gpllimit="max", prop="description", redirects=1):
        for p in r.get("query", {}).get("pages", []):
            if not p.get("missing"):
                out[p["title"]] = p.get("description", "")
    return out


def race_steps(rr, start, target, max_hops=12, sl=None, k=40):
    """Yield one event dict per step: start, hop (chosen link + top candidates), done.
    With a Shortlist, only its top-k links per page go to the reranker."""
    start, start_desc = resolve(start)
    target, target_desc = resolve(target)
    query = target + (f": {target_desc}" if target_desc else "")
    path, visited, current = [start], {start}, start
    t0 = time.time()
    yield {"type": "start", "start": start, "start_desc": start_desc, "target": target, "target_desc": target_desc}
    for hop in range(1, max_hops + 1):
        yield {"type": "thinking", "hop": hop, "page": current}
        ts = time.time()
        cands = links(current)
        t_fetch = time.time() - ts
        if target in cands:
            path.append(target)
            yield {"type": "hop", "hop": hop, "title": target, "desc": target_desc, "direct": True,
                   "n_links": len(cands), "fetch_s": t_fetch, "score_s": 0, "top": []}
            break
        options = [t for t in cands if t not in visited and t not in BOILERPLATE]
        if not options:
            yield {"type": "dead_end", "hop": hop, "page": current}
            break
        docs = [f"{t} — {cands[t]}" if cands[t] else t for t in options]
        n_links, t_short = len(options), 0.0
        if sl:
            ts = time.time()
            keep = sl.top(query, docs, k)
            options, docs = [options[i] for i in keep], [docs[i] for i in keep]
            t_short = time.time() - ts
        ts = time.time()
        scores = rr.score(INSTRUCTION, query, docs)
        ranked = sorted(zip(scores, options), reverse=True)
        current = ranked[0][1]
        yield {"type": "hop", "hop": hop, "title": current, "desc": cands[current], "direct": False,
               "n_links": n_links, "n_scored": len(options), "fetch_s": t_fetch,
               "shortlist_s": t_short, "score_s": time.time() - ts,
               "top": [{"title": t, "desc": cands[t], "score": sc} for sc, t in ranked[:8]]}
        path.append(current)
        visited.add(current)
    yield {"type": "done", "win": path[-1] == target, "hops": len(path) - 1, "secs": time.time() - t0, "path": path}


def race(rr, start, target, max_hops=12, sl=None):
    for e in race_steps(rr, start, target, max_hops, sl):
        if e["type"] == "start":
            print(f"\n=== {e['start']} -> {e['target']} ===")
        elif e["type"] == "hop" and e["direct"]:
            print(f"  {e['hop']:2d}. {e['title']}  (direct link)")
        elif e["type"] == "hop":
            runners = ", ".join(f"{c['title']} {c['score']:.2f}" for c in e["top"][1:3])
            print(f"  {e['hop']:2d}. {e['title']}  [{e['top'][0]['score']:.2f}]  {e['n_scored']} of {e['n_links']} links "
                  f"in {e['shortlist_s'] + e['score_s']:.1f}s  (next: {runners})")
        elif e["type"] == "dead_end":
            print("  dead end")
        elif e["type"] == "done":
            print(f"  -> {'WIN' if e['win'] else 'FAIL'} in {e['hops']} hops, {e['secs']:.0f}s")
            return e["win"], e["hops"], e["secs"]


if __name__ == "__main__":
    rr, sl = Reranker(batch_size=16), Shortlist()
    races = [tuple(sys.argv[1:3])] if len(sys.argv) == 3 else RACES
    results = [race(rr, s, t, sl=sl) for s, t in races]
    wins = [r for r in results if r[0]]
    print(f"\n{len(wins)}/{len(results)} wins", end="")
    if wins:
        print(f", median {sorted(h for _, h, _ in wins)[len(wins) // 2]} hops, "
              f"{sum(t for *_, t in results) / len(results):.0f}s avg per race")
