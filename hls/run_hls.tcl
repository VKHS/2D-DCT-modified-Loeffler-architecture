# ============================================================================
#  Vitis HLS flow for dct2d_modified_loeffler
#
#      cd <repository root>
#      vitis_hls -f hls/run_hls.tcl                 # csim + csynth
#      DCT_COSIM=1  vitis_hls -f hls/run_hls.tcl    # + C/RTL co-simulation
#      DCT_EXPORT=1 vitis_hls -f hls/run_hls.tcl    # + IP-catalog export
#
#  Part and clock can be changed with DCT_PART / DCT_CLK_NS.
# ============================================================================
set root [file normalize [file join [file dirname [info script]] ..]]
set part   [expr {[info exists ::env(DCT_PART)]   ? $::env(DCT_PART)   : "xc7a100tcsg324-1"}]
set clk_ns [expr {[info exists ::env(DCT_CLK_NS)] ? $::env(DCT_CLK_NS) : "5.0"}]

open_project -reset [file join $root build hls_dct2d]
set_top dct2d_modified_loeffler
add_files      [file join $root hls dct2d_loeffler.cpp]    -cflags "-I[file join $root hls]"
add_files -tb  [file join $root hls tb_dct2d_loeffler.cpp] -cflags "-I[file join $root hls]"

open_solution -reset artix7 -flow_target vivado
set_part $part
create_clock -period $clk_ns -name default

csim_design -argv [file join $root vectors]
csynth_design

if {[info exists ::env(DCT_COSIM)]} {
    cosim_design -argv [file join $root vectors] -trace_level port
}
if {[info exists ::env(DCT_EXPORT)]} {
    export_design -format ip_catalog -description "Modified-Loeffler 2-D DCT with TSC" -vendor "liu.se" -version "1.0"
}
exit
