# Timing constraint for out-of-context synthesis of dct2d_top.
# 4.831 ns = 207 MHz, the clock frequency reported in the paper for its
# VHDL implementation on Artix-7.  Relax or tighten as needed.
create_clock -period 4.831 -name clk [get_ports clk]
