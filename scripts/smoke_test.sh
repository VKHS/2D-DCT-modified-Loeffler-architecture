#!/usr/bin/env bash
# =============================================================================
#  One-command smoke test of every implementation against the golden vectors
#
#      bash scripts/smoke_test.sh
#
#  1  Python model           pytest (float/fixed engines, TSC, widths, vectors)
#  2  HLS C simulation       g++ build of hls/ + test bench on vectors/
#  3  Verilog RTL            Icarus Verilog: engine + 2-D processor, 2 modes x 2 input patterns
#  4  MATLAB / Octave model  matlab/smoke_test_matlab.m
#
#  A stage whose tool is not installed is reported as SKIPPED.
#  Exit status = number of failed stages.
# =============================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p build
pass=0; fail=0; skipped=0

stage() {
    local name="$1"; shift
    printf '\n=== %s\n' "$name"
    if "$@"; then
        printf -- '--- %s: PASS\n' "$name"; pass=$((pass + 1))
    else
        printf -- '--- %s: FAIL\n' "$name"; fail=$((fail + 1))
    fi
}
skip() {
    printf '\n=== %s\n--- %s: SKIPPED (%s not installed)\n' "$1" "$1" "$2"; skipped=$((skipped + 1))
}

if command -v python3 >/dev/null 2>&1 && python3 -c "import numpy, pytest" 2>/dev/null; then
    stage "Python model" python3 -m pytest -q python/tests
else
    skip "Python model" "python3 with numpy and pytest"
fi

if command -v g++ >/dev/null 2>&1; then
    stage "HLS C-simulation" bash -c 'g++ -std=c++17 -O2 -Wall -Wno-unknown-pragmas -Wno-unused-label -Ihls \
        hls/dct2d_loeffler.cpp hls/tb_dct2d_loeffler.cpp -o build/hls_csim && ./build/hls_csim vectors'
else
    skip "HLS C-simulation" "g++"
fi

if command -v iverilog >/dev/null 2>&1; then
    stage "Verilog RTL simulation" bash scripts/run_rtl_sim.sh
else
    skip "Verilog RTL simulation" "iverilog"
fi

if command -v octave >/dev/null 2>&1; then
    stage "MATLAB model (Octave)" octave --no-gui --quiet --eval "cd matlab; smoke_test_matlab"
elif command -v matlab >/dev/null 2>&1; then
    stage "MATLAB model" matlab -batch "cd matlab; smoke_test_matlab"
else
    skip "MATLAB model" "octave or matlab"
fi

printf '\n=== summary: %d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skipped"
exit "$fail"
