#!/usr/bin/env bash
# macOS double-click launcher — opens Terminal and starts Nine Men's Morris.
# To use: right-click → Open (first time only, to bypass Gatekeeper), then
# double-click from Finder on subsequent runs.
cd "$(dirname "$0")" && ./run_nmm.sh
