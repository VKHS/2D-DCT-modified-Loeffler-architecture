// ============================================================================
//  transpose_buffer.v
//  In-place 8x8 transpose memory with alternating orientation
//
//  A block written row-wise is read column-wise while the next block is
//  written column-wise into exactly the locations that are being read.  The
//  write orientation flips after every block, so a single 64-word register
//  file sustains one row per clock in and one column per clock out (no
//  ping-pong pair is needed).
//
//      orientation 0 : element (r, u) of the block at mem[8*r + u]
//      orientation 1 : element (r, u) of the block at mem[8*u + r]
//
//  Read is combinational (read-before-write in the same clock).
// ============================================================================
`timescale 1ns / 1ps
`default_nettype none

module transpose_buffer #(
    parameter integer W = 13
) (
    input  wire            clk,
    input  wire            rst,
    input  wire            wr_en,
    input  wire [2:0]      wr_row,
    input  wire [8*W-1:0]  wr_data,              // row r, element u at [u*W +: W]
    input  wire [2:0]      rd_col,
    output reg  [8*W-1:0]  rd_data,              // column u, element r at [r*W +: W]
    output reg             wr_orient,            // orientation of the block being written
    output reg             rd_orient             // orientation of the last completed block
);

    reg [W-1:0] mem [0:63];
    integer j, v;

    always @(posedge clk) begin
        if (rst) begin
            wr_orient <= 1'b0;
            rd_orient <= 1'b0;
        end else if (wr_en && (wr_row == 3'd7)) begin
            rd_orient <= wr_orient;
            wr_orient <= ~wr_orient;
        end
        if (wr_en) begin
            for (j = 0; j < 8; j = j + 1) begin
                if (wr_orient)
                    mem[j*8 + wr_row] <= wr_data[j*W +: W];
                else
                    mem[wr_row*8 + j] <= wr_data[j*W +: W];
            end
        end
    end

    always @(*) begin
        for (v = 0; v < 8; v = v + 1) begin
            if (rd_orient)
                rd_data[v*W +: W] = mem[rd_col*8 + v];
            else
                rd_data[v*W +: W] = mem[v*8 + rd_col];
        end
    end

endmodule

`default_nettype wire
