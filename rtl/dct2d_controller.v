// ============================================================================
//  dct2d_controller.v
//  Block sequencing of the 2-D DCT processor
//
//  * tags every input row with its index 0..7 (blocks arrive row by row)
//  * collects the TSC replacement mask of the block being written
//  * read FSM: when row 7 of a block enters the transpose buffer, the eight
//    columns of that block are issued to the column engine on the next eight
//    clocks.  A block that completes exactly when column 7 is issued starts
//    immediately, giving one block every 8 clocks.
//
//      S_IDLE --row 7 written--> S_READ (col 0..7) --col 7 & no new block--> S_IDLE
//                                   ^________row 7 written at col 7_______|
// ============================================================================
`timescale 1ns / 1ps
`default_nettype none

module dct2d_controller (
    input  wire       clk,
    input  wire       rst,
    // input rows
    input  wire       in_valid,
    output reg  [2:0] in_row,                    // index tag of the current input row
    // rows leaving the TSC unit (= transpose-buffer writes)
    input  wire       wr_valid,
    input  wire [2:0] wr_row,
    input  wire       wr_replaced,
    input  wire       wr_approx,
    // column issue
    output reg        rd_valid,
    output reg  [2:0] rd_col,
    output reg        rd_last,
    output reg        rd_approx,                 // stage-2 flag of the block being read
    output reg  [7:0] rd_mask,                   // replacement mask of the block being read
    output reg  [1:0] state
);

    localparam [1:0] S_IDLE = 2'd0,
                     S_READ = 2'd1;

    reg  [7:0] acc_mask;
    wire       blk_done = wr_valid && (wr_row == 3'd7);

    always @(posedge clk) begin
        if (rst) begin
            in_row    <= 3'd0;
            acc_mask  <= 8'd0;
            state     <= S_IDLE;
            rd_valid  <= 1'b0;
            rd_col    <= 3'd0;
            rd_last   <= 1'b0;
            rd_approx <= 1'b0;
            rd_mask   <= 8'd0;
        end else begin
            if (in_valid)
                in_row <= in_row + 3'd1;
            if (wr_valid)
                acc_mask[wr_row] <= wr_replaced;
            if (blk_done) begin
                state     <= S_READ;
                rd_valid  <= 1'b1;
                rd_col    <= 3'd0;
                rd_last   <= 1'b0;
                rd_approx <= wr_approx;
                rd_mask   <= {wr_replaced, acc_mask[6:0]};
            end else if (state == S_READ) begin
                if (rd_col == 3'd7) begin
                    state    <= S_IDLE;
                    rd_valid <= 1'b0;
                    rd_last  <= 1'b0;
                end else begin
                    rd_col  <= rd_col + 3'd1;
                    rd_last <= (rd_col == 3'd6);
                end
            end
        end
    end

endmodule

`default_nettype wire
