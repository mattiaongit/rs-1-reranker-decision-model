"""Record a flow in one of the local UIs as a GIF (used for the README).

Opens the page in the agent Chrome (./start_chrome.sh, port 9333), runs a JS snippet to start the
flow, screenshots the page until a JS condition is true, then builds an optimised GIF with ffmpeg.

    uv run tools/record_gif.py OUT.gif URL START_JS DONE_JS [--speed 2] [--height 900] [--hold 3]
"""
import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CDP = "http://127.0.0.1:9333"


async def record(url, start_js, done_js, frames_dir, width, height, interval, hold, timeout):
    req = urllib.request.Request(f"{CDP}/json/new?about:blank", method="PUT")
    tab = json.load(urllib.request.urlopen(req))
    shots = []
    try:
        async with websockets.connect(tab["webSocketDebuggerUrl"], max_size=None) as ws:
            n = 0

            async def send(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == n:
                        return m.get("result", {})

            async def ev(expr):
                r = await send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
                return r.get("result", {}).get("value")

            await send("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
            await send("Emulation.setFocusEmulationEnabled", enabled=True)
            await send("Page.navigate", url=url)
            await asyncio.sleep(2.0)
            await ev(start_js)
            started = time.time()
            done_at = None
            while time.time() - started < timeout:
                t = time.time()
                data = (await send("Page.captureScreenshot", format="jpeg", quality=85))["data"]
                path = frames_dir / f"{len(shots):05d}.jpg"
                path.write_bytes(base64.b64decode(data))
                shots.append((path, t))
                if done_at is None and await ev(done_js):
                    done_at = t
                if done_at and t - done_at >= hold:
                    break
                await asyncio.sleep(max(0, interval - (time.time() - t)))
    finally:
        urllib.request.urlopen(f"{CDP}/json/close/{tab['id']}")
    return shots, done_at is not None


def build_gif(shots, out, speed, gif_width, fps):
    lines = []
    for (path, t), nxt in zip(shots, shots[1:] + [None]):
        duration = ((nxt[1] - t) if nxt else 2.0) / speed
        lines.append(f"file '{path}'\nduration {duration:.3f}")
    lines.append(f"file '{shots[-1][0]}'")  # concat demuxer needs the last file twice
    listing = shots[0][0].parent / "list.txt"
    listing.write_text("\n".join(lines))
    vf = (f"fps={fps},scale={gif_width}:-1:flags=lanczos,split[a][b];"
          "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-vf", vf, "-loop", "0", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("url")
    ap.add_argument("start_js")
    ap.add_argument("done_js")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed-up")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--gif-width", type=int, default=960)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--interval", type=float, default=0.25, help="seconds between screenshots")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to keep recording after done")
    ap.add_argument("--timeout", type=float, default=240)
    a = ap.parse_args()
    frames = Path(tempfile.mkdtemp(prefix="rs1-frames-"))
    try:
        shots, finished = asyncio.run(record(a.url, a.start_js, a.done_js, frames, a.width, a.height,
                                             a.interval, a.hold, a.timeout))
        build_gif(shots, a.out, a.speed, a.gif_width, a.fps)
        size = Path(a.out).stat().st_size / 1e6
        print(f"{a.out}: {len(shots)} frames, finished={finished}, {size:.1f} MB")
    finally:
        shutil.rmtree(frames, ignore_errors=True)


if __name__ == "__main__":
    main()
