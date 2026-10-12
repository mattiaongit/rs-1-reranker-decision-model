"""Browser use: RS-1 plugged into jev-ultrafast's agent loop (third_party/jev-ultrafast, unmodified)."""
import os
import sys
from pathlib import Path

# The jev-ultrafast checkout (a git submodule); JEV_ULTRAFAST_DIR points elsewhere if needed.
JEV_ULTRAFAST = Path(os.environ.get("JEV_ULTRAFAST_DIR")
                     or Path(__file__).resolve().parents[3] / "third_party" / "jev-ultrafast")
if not (JEV_ULTRAFAST / "jev_ultrafast").is_dir():
    sys.exit(f"jev-ultrafast not found at {JEV_ULTRAFAST}. Run: git submodule update --init")
sys.path.insert(0, str(JEV_ULTRAFAST))
