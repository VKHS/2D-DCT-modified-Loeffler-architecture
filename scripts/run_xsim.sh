#!/usr/bin/env bash
# =============================================================================
#  Vivado simulator (xsim) regression; run from anywhere, works in the repo root
#  because the test benches read vectors/ with relative paths.
#
#      bash scripts/run_xsim.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
RTL="rtl/loeffler_dct8.v rtl/tsc_unit.v rtl/transpose_buffer.v rtl/dct2d_controller.v rtl/dct2d_top.v"

xvlog rtl/loeffler_dct8.v rtl/tb/tb_loeffler_dct8.v
xelab tb_loeffler_dct8 -s tb_engine
xsim tb_engine -R | grep -E '^(PASS|FAIL|MISMATCH)'

for mode in exact tsc; do
    define=""
    [ "$mode" = tsc ] && define="-d MODE_TSC"
    xvlog $define $RTL rtl/tb/tb_dct2d_top.v
    for gaps in 0 1; do
        xelab tb_dct2d_top -generic_top "GAPS=$gaps" -s "tb_${mode}_${gaps}"
        xsim "tb_${mode}_${gaps}" -R | grep -E '^(PASS|FAIL|MISMATCH|STATUS)'
    done
done
