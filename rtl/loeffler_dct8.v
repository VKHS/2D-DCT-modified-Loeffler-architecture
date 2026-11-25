// ============================================================================
//  loeffler_dct8.v
//  Fully pipelined 8-point modified Loeffler DCT engine
//
//  Y = sqrt(8) * C8 * x   (Loeffler scale), one 8-sample vector per clock
//  11 multipliers, 29 adders, every rotation in 3-multiplier form:
//      z = K*(xp+xq),  yp = z + A*xq,  yq = z - B*xp      (fused, rounded once)
//
//  Pipeline (register after every step, latency LAT = 7 clocks):
//      P1  stage 1 butterflies                          a0..a7
//      P2  stage 2 even butterflies, odd pre-adders     b0..b3, s47, s56
//      P3  odd products (6),  even Y0/Y4, pre-add s23    -> DSP M-register
//      P4  odd post-add+round b4..b7, even products (3)  -> DSP P/M-register
//      P5  stage 3 odd butterflies d4..d7, even Y2/Y6    (even half complete)
//      P6  stage 4 Y1/Y7, sqrt2 products (2)
//      P7  rounding of Y3/Y5, aligned output register    (odd half complete)
//
//  The even outputs (Y0,Y4,Y2,Y6) are complete after algorithmic stage 3 and
//  the odd outputs (Y1,Y7,Y3,Y5) after stage 4; P7 aligns both halves.
//
//  Word lengths: WD must hold every node; for 8-bit pixels with RF=2 guard
//  bits the row engine needs WI=10/WD=13 and the column engine WI=13/WD=16
//  (proved by python/loeffler2d/fixed_point.py::engine_ranges).
// ============================================================================
`timescale 1ns / 1ps
`default_nettype none

module loeffler_dct8 #(
    parameter integer WI   = 10,                 // input word width (signed)
    parameter integer WD   = 13,                 // data / output word width
    parameter integer CF   = 13,                 // constant fractional bits
    parameter integer WC   = 15,                 // constant width (signed)
    parameter integer TAGW = 1,                  // side-band width
    // rotation constants round(c * 2^CF); regenerate with
    // python/scripts/gen_rtl_constants.py when CF changes
    parameter signed [WC-1:0] E_K  =  4433,      // sqrt2*cos(6pi/16)
    parameter signed [WC-1:0] E_A  =  6270,      // sqrt2*(cos(2pi/16)-cos(6pi/16))
    parameter signed [WC-1:0] E_B  = 15137,      // sqrt2*(cos(2pi/16)+cos(6pi/16))
    parameter signed [WC-1:0] O3_K =  6811,      // cos(3pi/16)
    parameter signed [WC-1:0] O3_A = -2260,      // cos(5pi/16)-cos(3pi/16)
    parameter signed [WC-1:0] O3_B = 11363,      // cos(3pi/16)+cos(5pi/16)
    parameter signed [WC-1:0] O1_K =  8035,      // cos(pi/16)
    parameter signed [WC-1:0] O1_A = -6436,      // cos(7pi/16)-cos(pi/16)
    parameter signed [WC-1:0] O1_B =  9633,      // cos(pi/16)+cos(7pi/16)
    parameter signed [WC-1:0] R2   = 11585       // sqrt2
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              in_valid,
    input  wire [TAGW-1:0]   in_tag,
    input  wire [8*WI-1:0]   in_x,               // lane k: x[k] at [k*WI +: WI]
    output wire              out_valid,
    output wire [TAGW-1:0]   out_tag,
    output wire [8*WD-1:0]   out_y               // lane k: Y[k] at [k*WD +: WD]
);

    localparam integer WP  = WD + WC;            // product width
    localparam integer LAT = 7;                  // pipeline latency
    localparam signed [WP:0] RND = 1 <<< (CF - 1);

    // ------------------------------------------------------------- input lanes
    wire signed [WD-1:0] x0 = $signed(in_x[0*WI +: WI]);
    wire signed [WD-1:0] x1 = $signed(in_x[1*WI +: WI]);
    wire signed [WD-1:0] x2 = $signed(in_x[2*WI +: WI]);
    wire signed [WD-1:0] x3 = $signed(in_x[3*WI +: WI]);
    wire signed [WD-1:0] x4 = $signed(in_x[4*WI +: WI]);
    wire signed [WD-1:0] x5 = $signed(in_x[5*WI +: WI]);
    wire signed [WD-1:0] x6 = $signed(in_x[6*WI +: WI]);
    wire signed [WD-1:0] x7 = $signed(in_x[7*WI +: WI]);

    // ------------------------------------------------- P1: stage-1 butterflies
    reg signed [WD-1:0] a0, a1, a2, a3, a4, a5, a6, a7;
    always @(posedge clk) begin
        a0 <= x0 + x7;   a7 <= x0 - x7;
        a1 <= x1 + x6;   a6 <= x1 - x6;
        a2 <= x2 + x5;   a5 <= x2 - x5;
        a3 <= x3 + x4;   a4 <= x3 - x4;
    end

    // ------------------------------ P2: stage-2 even butterflies, odd pre-adds
    reg signed [WD-1:0] b0, b1, b2, b3, s47, s56, a4_2, a5_2, a6_2, a7_2;
    always @(posedge clk) begin
        b0  <= a0 + a3;  b3  <= a0 - a3;
        b1  <= a1 + a2;  b2  <= a1 - a2;
        s47 <= a4 + a7;  s56 <= a5 + a6;
        a4_2 <= a4;  a5_2 <= a5;  a6_2 <= a6;  a7_2 <= a7;
    end

    // ------------------------- P3: odd rotation products, even Y0/Y4, pre-add
    reg signed [WP-1:0] p_o3k, p_o3a, p_o3b, p_o1k, p_o1a, p_o1b;
    reg signed [WD-1:0] y0_3, y4_3, s23, b2_3, b3_3;
    always @(posedge clk) begin
        p_o3k <= O3_K * s47;     // rotation 3pi/16 on (a4, a7)
        p_o3a <= O3_A * a7_2;
        p_o3b <= O3_B * a4_2;
        p_o1k <= O1_K * s56;     // rotation  pi/16 on (a5, a6)
        p_o1a <= O1_A * a6_2;
        p_o1b <= O1_B * a5_2;
        y0_3  <= b0 + b1;        // DC
        y4_3  <= b0 - b1;
        s23   <= b2 + b3;
        b2_3  <= b2;
        b3_3  <= b3;
    end

    // -------------- P4: odd post-add + rounding (b4..b7), even rotation products
    wire signed [WP:0] sum_b4 = p_o3k + p_o3a + RND;
    wire signed [WP:0] sum_b7 = p_o3k - p_o3b + RND;
    wire signed [WP:0] sum_b5 = p_o1k + p_o1a + RND;
    wire signed [WP:0] sum_b6 = p_o1k - p_o1b + RND;
    reg  signed [WD-1:0] b4, b5, b6, b7, y0_4, y4_4;
    reg  signed [WP-1:0] p_ek, p_ea, p_eb;
    always @(posedge clk) begin
        b4   <= sum_b4 >>> CF;
        b7   <= sum_b7 >>> CF;
        b5   <= sum_b5 >>> CF;
        b6   <= sum_b6 >>> CF;
        p_ek <= E_K * s23;       // sqrt2-scaled rotation 6pi/16 on (b2, b3)
        p_ea <= E_A * b3_3;
        p_eb <= E_B * b2_3;
        y0_4 <= y0_3;
        y4_4 <= y4_3;
    end

    // ------------------- P5: stage-3 odd butterflies, even Y2/Y6 (even done)
    wire signed [WP:0] sum_y2 = p_ek + p_ea + RND;
    wire signed [WP:0] sum_y6 = p_ek - p_eb + RND;
    reg  signed [WD-1:0] d4, d5, d6, d7, y0_5, y4_5, y2_5, y6_5;
    always @(posedge clk) begin
        d4   <= b4 + b6;   d6 <= b4 - b6;
        d7   <= b7 + b5;   d5 <= b7 - b5;
        y2_5 <= sum_y2 >>> CF;
        y6_5 <= sum_y6 >>> CF;
        y0_5 <= y0_4;
        y4_5 <= y4_4;
    end

    // ---------------------------------- P6: stage-4 Y1/Y7, sqrt2 products
    reg signed [WD-1:0] y1_6, y7_6, y0_6, y4_6, y2_6, y6_6;
    reg signed [WP-1:0] p_r3, p_r5;
    always @(posedge clk) begin
        y1_6 <= d7 + d4;
        y7_6 <= d7 - d4;
        p_r3 <= R2 * d5;
        p_r5 <= R2 * d6;
        y0_6 <= y0_5;  y4_6 <= y4_5;  y2_6 <= y2_5;  y6_6 <= y6_5;
    end

    // ------------------------------- P7: rounding of Y3/Y5, output register
    wire signed [WP:0] sum_y3 = p_r3 + RND;
    wire signed [WP:0] sum_y5 = p_r5 + RND;
    reg  signed [WD-1:0] o0, o1, o2, o3, o4, o5, o6, o7;
    always @(posedge clk) begin
        o0 <= y0_6;
        o1 <= y1_6;
        o2 <= y2_6;
        o3 <= sum_y3 >>> CF;
        o4 <= y4_6;
        o5 <= sum_y5 >>> CF;
        o6 <= y6_6;
        o7 <= y7_6;
    end
    assign out_y = {o7, o6, o5, o4, o3, o2, o1, o0};

    // ------------------------------------------------ valid / side-band pipe
    reg [LAT-1:0]      vpipe;
    reg [LAT*TAGW-1:0] tpipe;
    always @(posedge clk) begin
        if (rst) vpipe <= {LAT{1'b0}};
        else     vpipe <= {vpipe[LAT-2:0], in_valid};
        tpipe <= {tpipe[(LAT-1)*TAGW-1:0], in_tag};
    end
    assign out_valid = vpipe[LAT-1];
    assign out_tag   = tpipe[LAT*TAGW-1 -: TAGW];

endmodule

`default_nettype wire
