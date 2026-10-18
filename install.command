#!/usr/bin/env bash
# macOS double-click installer — opens Terminal and runs install.sh.
# Right-click → Open the first time to bypass Gatekeeper, then
# double-click from Finder on subsequent runs.
cd "$(dirname "$0")" && ./install.sh
