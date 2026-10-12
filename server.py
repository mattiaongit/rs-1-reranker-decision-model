"""Wikirace UI: http://localhost:8804 — streams each hop over Server-Sent Events.

    uv run server.py
"""
import json
import os
import threading
import time

# Cap PyTorch's GPU memory (unified RAM on Apple silicon) at ~30% of the recommended
# working set: an oversized batch fails with an error instead of pushing macOS into swap.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")  # must not exceed the high ratio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from reranker import Reranker
from mcq import INSTRUCTION as MCQ_INSTRUCTION, QUESTIONS
from shortlist import Shortlist
from wikirace import race_steps

PORT = 8804
ROOT = Path(__file__).parent
rr = Reranker(batch_size=16)
sl = Shortlist()
gpu = threading.Lock()  # one race at a time: the model isn't shared across threads


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send_body((ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/mcq":
            self.send_body((ROOT / "mcq.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/mcq/presets":
            self.send_json({"questions": QUESTIONS, "instruction": MCQ_INSTRUCTION})
        elif url.path == "/race":
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.stream(q.get("start", ""), q.get("target", ""), min(int(q.get("max_hops", 30)), 100), int(q.get("k", 40)))
        else:
            self.send_error(404)

    def do_POST(self):
        """/score  {"instruction", "query", "docs": [...]}  ->  {"scores": [...], "secs"}"""
        if urlparse(self.path).path != "/score":
            return self.send_error(404)
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
        docs = [str(d) for d in req.get("docs", [])][:50]
        with gpu:
            t = time.time()
            scores = rr.score(req.get("instruction") or MCQ_INSTRUCTION, str(req.get("query", "")), docs) if docs else []
        self.send_json({"scores": scores, "secs": time.time() - t})

    def send_body(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj):
        self.send_body(json.dumps(obj).encode(), "application/json")

    def send_event(self, e):
        self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
        self.wfile.flush()

    def stream(self, start, target, max_hops, k):
        if not gpu.acquire(blocking=False):
            self.send_event({"type": "busy"})
            gpu.acquire()
        try:
            for e in race_steps(rr, start, target, max_hops, sl=sl if k > 0 else None, k=k):
                self.send_event(e)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser closed the stream: stop racing
        except Exception as ex:
            try:
                self.send_event({"type": "error", "message": str(ex)})
            except OSError:
                pass
        finally:
            gpu.release()


if __name__ == "__main__":
    print(f"Wikirace UI on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
