#!/usr/bin/env bash
# =============================================================================
#  Icarus Verilog regression of rtl/ against vectors/
#
#      bash scripts/run_rtl_sim.sh            # 1 engine run + 4 processor runs
#      DCT_VCD=1 bash scripts/run_rtl_sim.sh  # also writes build/tb_dct2d_top.vcd
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
RTL="rtl/loeffler_dct8.v rtl/tsc_unit.v rtl/transpose_buffer.v rtl/dct2d_controller.v rtl/dct2d_top.v"
status=0

nblocks() { awk -F' = ' '$1 == "n_blocks" {print $2}' "vectors/$1/config.txt"; }
check() {
    cat "$1"
    grep -q '^PASS' "$1" || status=1
}

iverilog -g2005 -o build/tb_engine.vvp rtl/tb/tb_loeffler_dct8.v rtl/loeffler_dct8.v
vvp -n build/tb_engine.vvp > build/tb_engine.log
check build/tb_engine.log

for mode in exact tsc; do
    define=""
    [ "$mode" = tsc ] && define="-DMODE_TSC"
    for gaps in 0 1; do
        bin="build/tb_top_${mode}_gaps${gaps}.vvp"
        iverilog -g2005 $define -Ptb_dct2d_top.NB="$(nblocks "$mode")" -Ptb_dct2d_top.GAPS="$gaps" \
            -o "$bin" rtl/tb/tb_dct2d_top.v $RTL
        vvp -n "$bin" > "${bin%.vvp}.log"
        check "${bin%.vvp}.log"
    done
done

if [ "${DCT_VCD:-0}" = 1 ]; then
    vvp -n build/tb_top_tsc_gaps0.vvp +vcd > /dev/null
    echo "waveform: build/tb_dct2d_top.vcd"
fi
exit "$status"
