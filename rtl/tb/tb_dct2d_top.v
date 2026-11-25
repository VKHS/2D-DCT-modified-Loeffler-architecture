// ============================================================================
//  tb_dct2d_top.v  -  self-checking test bench of the 2-D DCT processor
//
//  Golden vectors: vectors/<mode>/ (python/scripts/gen_vectors.py)
//      iverilog -g2005 -o tb.vvp [-DMODE_TSC] [-Ptb_dct2d_top.GAPS=1] \
//               rtl/tb/tb_dct2d_top.v rtl/*.v
//      vvp tb.vvp [+vcd]                  (run from the repository root)
//
//  MODE_TSC undefined : TSC and approximation off  -> exact fixed-point DCT
//  MODE_TSC defined   : TSC on (t0=1, step=1, tmax=10, min_sim=3), zone 5
//  GAPS=1             : random idle cycles between input rows
// ============================================================================
`timescale 1ns / 1ps
`default_nettype none

`ifdef MODE_TSC
  `define VEC_ROWS   "vectors/tsc/input_rows.hex"
  `define VEC_COLS   "vectors/tsc/expected_cols.hex"
  `define VEC_STATUS "vectors/tsc/expected_status.hex"
  `define CFG_TSC_EN 1'b1
  `define CFG_APPROX 2'd1
  `define MODE_NAME  "tsc"
`else
  `define VEC_ROWS   "vectors/exact/input_rows.hex"
  `define VEC_COLS   "vectors/exact/expected_cols.hex"
  `define VEC_STATUS "vectors/exact/expected_status.hex"
  `define CFG_TSC_EN 1'b0
  `define CFG_APPROX 2'd0
  `define MODE_NAME  "exact"
`endif

module tb_dct2d_top;

    parameter integer NB   = 63;                 // blocks in the vector files
    parameter integer GAPS = 0;                  // 1: random input bubbles

    reg clk = 1'b0;
    always #5 clk = ~clk;                        // 100 MHz simulation clock

    reg         rst;
    reg         in_valid;
    reg  [63:0] in_row;
    wire        out_valid, out_last, out_approx;
    wire [2:0]  out_col;
    wire [95:0] out_coef;
    wire [7:0]  out_replaced;

    dct2d_top dut (
        .clk(clk), .rst(rst),
        .cfg_tsc_en(`CFG_TSC_EN), .cfg_t0(4'd1), .cfg_step(4'd1), .cfg_tmax(4'd10),
        .cfg_min_sim(2'd3), .cfg_approx_mode(`CFG_APPROX), .cfg_zone(4'd5),
        .in_valid(in_valid), .in_row(in_row),
        .out_valid(out_valid), .out_col(out_col), .out_last(out_last),
        .out_coef(out_coef), .out_replaced(out_replaced), .out_approx(out_approx)
    );

    reg [63:0] rows   [0:8*NB-1];
    reg [95:0] cols   [0:8*NB-1];
    reg [8:0]  status [0:NB-1];

    integer i, ncol, nblk, errors, seed, first_out, cycle;

    // ------------------------------------------------------------ stimulus
    initial begin
        $readmemh(`VEC_ROWS, rows);
        $readmemh(`VEC_COLS, cols);
        $readmemh(`VEC_STATUS, status);
        if ($test$plusargs("vcd")) begin
            $dumpfile("build/tb_dct2d_top.vcd");
            $dumpvars(0, tb_dct2d_top);
        end
        seed = 2019;
        rst = 1'b1;
        in_valid = 1'b0;
        in_row = 64'd0;
        repeat (3) @(negedge clk);
        rst = 1'b0;
        i = 0;
        while (i < 8*NB) begin
            @(negedge clk);
            if ((GAPS != 0) && (($random(seed) & 3) == 0)) begin
                in_valid = 1'b0;
            end else begin
                in_valid = 1'b1;
                in_row   = rows[i];
                i = i + 1;
            end
        end
        @(negedge clk);
        in_valid = 1'b0;
    end

    // ------------------------------------------------------------ checker
    initial begin
        errors = 0; ncol = 0; nblk = 0; first_out = -1; cycle = 0;
    end

    always @(posedge clk) begin
        cycle = cycle + 1;
        if (!rst && out_valid) begin
            if (first_out < 0) first_out = cycle;
            if (out_coef !== cols[ncol]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("MISMATCH block %0d column %0d: got %h expected %h",
                             ncol / 8, ncol % 8, out_coef, cols[ncol]);
            end
            if (out_col !== ncol[2:0]) begin
                errors = errors + 1;
                $display("ORDER ERROR: column index %0d, expected %0d", out_col, ncol % 8);
            end
            if (out_last) begin
                if ({out_approx, out_replaced} !== status[nblk]) begin
                    errors = errors + 1;
                    $display("STATUS MISMATCH block %0d: got %h expected %h",
                             nblk, {out_approx, out_replaced}, status[nblk]);
                end
                nblk = nblk + 1;
            end
            ncol = ncol + 1;
            if (ncol == 8*NB) begin
                if (errors == 0)
                    $display("PASS: dct2d_top mode=%s gaps=%0d blocks=%0d columns=%0d first_output_cycle=%0d",
                             `MODE_NAME, GAPS, NB, ncol, first_out);
                else
                    $display("FAIL: dct2d_top mode=%s gaps=%0d errors=%0d", `MODE_NAME, GAPS, errors);
                $finish;
            end
        end
    end

    initial begin
        #(10 * (40 * NB + 1000));
        $display("FAIL: timeout after %0d columns", ncol);
        $finish;
    end

endmodule

`default_nettype wire
