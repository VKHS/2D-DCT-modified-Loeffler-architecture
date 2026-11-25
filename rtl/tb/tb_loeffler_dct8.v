// ============================================================================
//  tb_loeffler_dct8.v  -  self-checking test bench of the 1-D engine
//
//      iverilog -g2005 -o tb_engine.vvp rtl/tb/tb_loeffler_dct8.v rtl/loeffler_dct8.v
//      vvp tb_engine.vvp                    (run from the repository root)
//
//  vectors/engine/input.hex     8 x 10-bit lanes per line (row-engine range)
//  vectors/engine/expected.hex  8 x 13-bit lanes per line (bit-exact model)
// ============================================================================
`timescale 1ns / 1ps
`default_nettype none

module tb_loeffler_dct8;

    parameter integer NV = 512;
    localparam integer WI = 10, WD = 13;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg              rst;
    reg              in_valid;
    reg  [8*WI-1:0]  in_x;
    wire             out_valid;
    wire [0:0]       out_tag;
    wire [8*WD-1:0]  out_y;

    loeffler_dct8 #(.WI(WI), .WD(WD), .CF(13), .TAGW(1)) dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_tag(1'b0), .in_x(in_x),
        .out_valid(out_valid), .out_tag(out_tag), .out_y(out_y)
    );

    reg [8*WI-1:0] xin [0:NV-1];
    reg [8*WD-1:0] expv [0:NV-1];
    integer i, n, errors, lat, t_in;

    initial begin
        $readmemh("vectors/engine/input.hex", xin);
        $readmemh("vectors/engine/expected.hex", expv);
        rst = 1'b1; in_valid = 1'b0; in_x = 0;
        repeat (2) @(negedge clk);
        rst = 1'b0;
        for (i = 0; i < NV; i = i + 1) begin
            @(negedge clk);
            in_valid = 1'b1;
            in_x = xin[i];
            if (i == 0) t_in = $time;
        end
        @(negedge clk);
        in_valid = 1'b0;
    end

    initial begin
        n = 0; errors = 0; lat = -1;
    end

    always @(posedge clk) begin
        if (!rst && out_valid) begin
            if (lat < 0) lat = ($time - t_in - 5) / 10;
            if (out_y !== expv[n]) begin
                errors = errors + 1;
                if (errors <= 10) $display("MISMATCH vector %0d: got %h expected %h", n, out_y, expv[n]);
            end
            n = n + 1;
            if (n == NV) begin
                if (errors == 0) $display("PASS: loeffler_dct8 %0d vectors, latency %0d clocks", NV, lat);
                else             $display("FAIL: loeffler_dct8 %0d errors", errors);
                $finish;
            end
        end
    end

    initial begin
        #(10 * (NV + 100));
        $display("FAIL: timeout");
        $finish;
    end

endmodule

`default_nettype wire
