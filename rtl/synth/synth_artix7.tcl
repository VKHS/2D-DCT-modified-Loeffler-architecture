# ============================================================================
#  Vivado non-project flow: out-of-context synthesis and implementation of
#  dct2d_top on Artix-7, with utilisation and timing reports.
#
#      cd <repository root>
#      vivado -mode batch -source rtl/synth/synth_artix7.tcl
#      DCT_PART=xc7a35tcpg236-1 vivado -mode batch -source rtl/synth/synth_artix7.tcl
#
#  Reports are written to build/vivado/.
# ============================================================================
set root [file normalize [file join [file dirname [info script]] .. ..]]
set part [expr {[info exists ::env(DCT_PART)] ? $::env(DCT_PART) : "xc7a100tcsg324-1"}]
set out  [file join $root build vivado]
file mkdir $out

read_verilog [list \
    [file join $root rtl loeffler_dct8.v] \
    [file join $root rtl tsc_unit.v] \
    [file join $root rtl transpose_buffer.v] \
    [file join $root rtl dct2d_controller.v] \
    [file join $root rtl dct2d_top.v]]
read_xdc [file join $root rtl synth dct2d_top.xdc]

synth_design -top dct2d_top -part $part -mode out_of_context -flatten_hierarchy rebuilt
report_utilization -hierarchical -file [file join $out utilization_synth.rpt]
opt_design
place_design
phys_opt_design
route_design
report_utilization -file [file join $out utilization_impl.rpt]
report_utilization -hierarchical -file [file join $out utilization_hier.rpt]
report_timing_summary -max_paths 10 -file [file join $out timing_summary.rpt]
report_power -file [file join $out power.rpt]
write_checkpoint -force [file join $out dct2d_top_routed.dcp]
puts "Reports in $out"
