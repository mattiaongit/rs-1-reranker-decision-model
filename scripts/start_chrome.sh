#!/bin/sh
# Separate Chrome for the agent: own empty profile, remote debugging on 9333. Your normal Chrome is untouched.
open -na "Google Chrome" --args --user-data-dir="$(cd "$(dirname "$0")/.." && pwd)/.chrome-agent" --remote-debugging-port=9333 \
  --no-first-run --no-default-browser-check about:blank
