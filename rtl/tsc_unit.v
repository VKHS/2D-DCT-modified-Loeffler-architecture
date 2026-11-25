// ============================================================================
//  tsc_unit.v
//  Threshold Set and Comparison (TSC) unit
//
//  Stage 1  The DCT of row r is compared with the spectrum kept for row r-1
//           (sum of absolute differences).  SAD < TINT[m] -> row r is
//           replaced by its predecessor.  After every dissimilar pair the
//           threshold index m grows by cfg_step (0.001 per step) up to cfg_tmax.
//  Stage 2  When at least cfg_min_sim of the pairs (1,0),(2,1),(3,2) are
//           similar, the block is marked for column-wise approximation.
//
//  One row per clock is sustained by a speculative dual-reference SAD:
//      sad_a = SAD(Y_r, Y_{r-1})   correct if row r-1 was kept
//      sad_b = SAD(Y_r, ref)       correct if row r-1 was replaced
//  and the decision of row r-1 selects the valid one in the next cycle, so
//  the loop "decision -> reference -> SAD" is reduced to a 2:1 mux and one
//  comparator.  Valid gaps between rows are handled by the same selection.
//
//  Latency: 2 clocks.  Threshold table TINT[m] = round(m*0.001*255*8*sqrt8*2^RF)
//  (default: RF = 2), entry m at bits [10*m +: 10].
// ============================================================================
`timescale 1ns / 1ps
`default_nettype none

module tsc_unit #(
    parameter integer W = 13,                    // row-spectrum word width
    parameter [159:0] TINT = {10'd346, 10'd323, 10'd300, 10'd277,
                              10'd254, 10'd231, 10'd208, 10'd185,
                              10'd162, 10'd138, 10'd115, 10'd92,
                              10'd69,  10'd46,  10'd23,  10'd0}
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              cfg_enable,
    input  wire [3:0]        cfg_t0,             // default threshold index
    input  wire [3:0]        cfg_step,           // adaptive increment
    input  wire [3:0]        cfg_tmax,           // maximum threshold index
    input  wire [1:0]        cfg_min_sim,        // stage-2 trigger (0..3)
    input  wire [1:0]        cfg_approx_mode,    // 0 off, 1 TSC, 2 always
    input  wire              in_valid,
    input  wire [2:0]        in_row,
    input  wire [8*W-1:0]    in_y,
    output reg               out_valid,
    output reg  [2:0]        out_row,
    output reg  [8*W-1:0]    out_y,
    output reg               out_replaced,
    output reg               out_approx,         // stage-2 flag, final from row 3 on
    output reg  [3:0]        out_tidx            // threshold index applied to this row
);

    wire [159:0] tint_tab = TINT;

    // ------------------------------------------------------------ stage A
    reg  [8*W-1:0] prev_raw;                     // Y_{r-1} as received
    reg  [8*W-1:0] ref_y;                        // spectrum kept for the last decided row
    reg  [W+3:0]   sad_a_c, sad_b_c;
    reg  signed [W:0] da, db;
    reg  [W:0]     ma, mb;
    integer k;
    always @(*) begin
        sad_a_c = {(W+4){1'b0}};
        sad_b_c = {(W+4){1'b0}};
        for (k = 0; k < 8; k = k + 1) begin
            da = $signed(in_y[k*W +: W]) - $signed(prev_raw[k*W +: W]);
            db = $signed(in_y[k*W +: W]) - $signed(ref_y[k*W +: W]);
            ma = da[W] ? -da : da;
            mb = db[W] ? -db : db;
            sad_a_c = sad_a_c + ma;
            sad_b_c = sad_b_c + mb;
        end
    end

    reg            va;
    reg  [2:0]     row_a;
    reg  [8*W-1:0] y_a;
    reg  [W+3:0]   sad_a, sad_b;
    always @(posedge clk) begin
        if (rst) va <= 1'b0;
        else     va <= in_valid;
        if (in_valid) begin
            row_a    <= in_row;
            y_a      <= in_y;
            sad_a    <= sad_a_c;
            sad_b    <= sad_b_c;
            prev_raw <= in_y;
        end
    end

    // ------------------------------------------------------------ stage B
    reg            last_rep;                     // decision of the previous row
    reg  [3:0]     tidx;                         // current threshold index
    reg  [1:0]     nsim;                         // similar pairs among rows 1..2
    wire [W+3:0]   sad_sel  = last_rep ? sad_b : sad_a;
    wire [9:0]     tint_cur = tint_tab[tidx*10 +: 10];
    wire           similar  = cfg_enable && (row_a != 3'd0) && (sad_sel < tint_cur);
    wire [4:0]     tinc     = tidx + cfg_step;
    wire [3:0]     tnext    = (tinc > cfg_tmax) ? cfg_tmax : tinc[3:0];
    wire [1:0]     nsim_tot = nsim + {1'b0, similar};
    wire           approx_now = (cfg_approx_mode == 2'd2) ||
                                ((cfg_approx_mode == 2'd1) && cfg_enable && (nsim_tot >= cfg_min_sim));

    always @(posedge clk) begin
        if (rst) begin
            out_valid  <= 1'b0;
            last_rep   <= 1'b0;
            tidx       <= 4'd0;
            nsim       <= 2'd0;
            out_approx <= 1'b0;
        end else begin
            out_valid <= va;
            if (va) begin
                out_row      <= row_a;
                out_replaced <= similar;
                out_y        <= similar ? ref_y : y_a;
                out_tidx     <= (row_a == 3'd0) ? cfg_t0 : tidx;
                last_rep     <= similar;
                if (!similar)
                    ref_y <= y_a;
                if (row_a == 3'd0) begin
                    tidx       <= cfg_t0;
                    nsim       <= 2'd0;
                    out_approx <= 1'b0;
                end else begin
                    if (!similar)
                        tidx <= tnext;
                    if (row_a <= 3'd2)
                        nsim <= nsim_tot;
                    if (row_a == 3'd3)
                        out_approx <= approx_now;
                end
            end
        end
    end

endmodule

`default_nettype wire
