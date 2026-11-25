// ============================================================================
//  dct2d_top.v
//  High-speed 8x8 2-D DCT processor based on the modified Loeffler engine
//
//    pixels --> level shift --> ROW ENGINE --> TSC --> TRANSPOSE --> COLUMN ENGINE
//     8/clk     (p-128)<<RF     (Loeffler)    unit     in-place      (Loeffler)
//                                                                       |
//    F[v][u] <-- triangular zone (stage 2) <-- (Z + 2^(RF+2)) >>> (RF+3) <--+
//
//  Throughput : one 8x8 block every 8 clocks (one row in, one column out per clock)
//  Latency    : 18 clocks from the last row of a block to its first column
//  Output     : column u of the orthonormal 2-D DCT per clock,
//               F[v][u] (12-bit two's complement) at out_coef[12*v +: 12]
//
//  Configuration inputs are quasi-static (change them between frames).
// ============================================================================
`timescale 1ns / 1ps
`default_nettype none

module dct2d_top #(
    parameter integer CF = 13,                   // constant fractional bits
    parameter integer RF = 2,                    // row-to-column guard bits
    parameter integer WR = 13,                   // row-engine data width (8+RF+3)
    parameter integer WK = 16,                   // column-engine data width (WR+3)
    parameter integer WO = 12                    // output coefficient width
) (
    input  wire            clk,
    input  wire            rst,
    // configuration
    input  wire            cfg_tsc_en,
    input  wire [3:0]      cfg_t0,
    input  wire [3:0]      cfg_step,
    input  wire [3:0]      cfg_tmax,
    input  wire [1:0]      cfg_min_sim,
    input  wire [1:0]      cfg_approx_mode,      // 0 off, 1 TSC-triggered, 2 always
    input  wire [3:0]      cfg_zone,             // keep u + v <= zone
    // pixel rows
    input  wire            in_valid,
    input  wire [63:0]     in_row,               // pixel k at [8k +: 8]
    // coefficient columns
    output reg             out_valid,
    output reg  [2:0]      out_col,              // horizontal frequency u
    output reg             out_last,             // column 7 of a block
    output reg  [8*WO-1:0] out_coef,             // F[v][u] at [WO*v +: WO]
    output reg  [7:0]      out_replaced,         // TSC mask of the block (with out_last)
    output reg             out_approx            // stage-2 flag of the block (with out_last)
);

    localparam integer WI = 8 + RF;              // row-engine input width

    // -------------------------------------------- level shift and guard bits
    wire [8*WI-1:0] row_x;
    assign row_x[0*WI +: WI] = {~in_row[ 7], in_row[ 6: 0], {RF{1'b0}}};
    assign row_x[1*WI +: WI] = {~in_row[15], in_row[14: 8], {RF{1'b0}}};
    assign row_x[2*WI +: WI] = {~in_row[23], in_row[22:16], {RF{1'b0}}};
    assign row_x[3*WI +: WI] = {~in_row[31], in_row[30:24], {RF{1'b0}}};
    assign row_x[4*WI +: WI] = {~in_row[39], in_row[38:32], {RF{1'b0}}};
    assign row_x[5*WI +: WI] = {~in_row[47], in_row[46:40], {RF{1'b0}}};
    assign row_x[6*WI +: WI] = {~in_row[55], in_row[54:48], {RF{1'b0}}};
    assign row_x[7*WI +: WI] = {~in_row[63], in_row[62:56], {RF{1'b0}}};

    // -------------------------------------------------------- controller
    wire [2:0] ctl_in_row;
    wire       ctl_rd_valid, ctl_rd_last, ctl_rd_approx;
    wire [2:0] ctl_rd_col;
    wire [7:0] ctl_rd_mask;
    wire [1:0] ctl_state;

    // -------------------------------------------------------- row engine
    wire            row_v;
    wire [2:0]      row_tag;
    wire [8*WR-1:0] row_y;

    loeffler_dct8 #(.WI(WI), .WD(WR), .CF(CF), .TAGW(3)) u_row_engine (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_tag(ctl_in_row), .in_x(row_x),
        .out_valid(row_v), .out_tag(row_tag), .out_y(row_y)
    );

    // -------------------------------------------------------- TSC unit
    wire            tsc_v, tsc_rep, tsc_approx;
    wire [2:0]      tsc_row;
    wire [8*WR-1:0] tsc_y;
    wire [3:0]      tsc_tidx;

    tsc_unit #(.W(WR)) u_tsc (
        .clk(clk), .rst(rst),
        .cfg_enable(cfg_tsc_en), .cfg_t0(cfg_t0), .cfg_step(cfg_step), .cfg_tmax(cfg_tmax),
        .cfg_min_sim(cfg_min_sim), .cfg_approx_mode(cfg_approx_mode),
        .in_valid(row_v), .in_row(row_tag), .in_y(row_y),
        .out_valid(tsc_v), .out_row(tsc_row), .out_y(tsc_y),
        .out_replaced(tsc_rep), .out_approx(tsc_approx), .out_tidx(tsc_tidx)
    );

    dct2d_controller u_ctrl (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_row(ctl_in_row),
        .wr_valid(tsc_v), .wr_row(tsc_row), .wr_replaced(tsc_rep), .wr_approx(tsc_approx),
        .rd_valid(ctl_rd_valid), .rd_col(ctl_rd_col), .rd_last(ctl_rd_last),
        .rd_approx(ctl_rd_approx), .rd_mask(ctl_rd_mask), .state(ctl_state)
    );

    // -------------------------------------------------------- transpose
    wire [8*WR-1:0] col_x;
    wire            tb_wr_orient, tb_rd_orient;

    transpose_buffer #(.W(WR)) u_transpose (
        .clk(clk), .rst(rst),
        .wr_en(tsc_v), .wr_row(tsc_row), .wr_data(tsc_y),
        .rd_col(ctl_rd_col), .rd_data(col_x),
        .wr_orient(tb_wr_orient), .rd_orient(tb_rd_orient)
    );

    // -------------------------------------------------------- column engine
    wire            col_v;
    wire [12:0]     col_tag;
    wire [8*WK-1:0] col_z;

    loeffler_dct8 #(.WI(WR), .WD(WK), .CF(CF), .TAGW(13)) u_col_engine (
        .clk(clk), .rst(rst),
        .in_valid(ctl_rd_valid),
        .in_tag({ctl_rd_approx, ctl_rd_last, ctl_rd_col, ctl_rd_mask}),
        .in_x(col_x),
        .out_valid(col_v), .out_tag(col_tag), .out_y(col_z)
    );

    // ---------------------- final normalisation and triangular approximation
    wire       t_approx = col_tag[12];
    wire       t_last   = col_tag[11];
    wire [2:0] t_col    = col_tag[10:8];
    wire [7:0] t_mask   = col_tag[7:0];

    reg [8*WO-1:0]   coef_c;
    reg signed [WK:0] zr;
    integer v;
    always @(*) begin
        for (v = 0; v < 8; v = v + 1) begin
            zr = $signed(col_z[v*WK +: WK]) + (1 <<< (RF + 2));
            if (t_approx && ((t_col + v) > cfg_zone))
                coef_c[v*WO +: WO] = {WO{1'b0}};
            else
                coef_c[v*WO +: WO] = zr >>> (RF + 3);
        end
    end

    always @(posedge clk) begin
        if (rst) out_valid <= 1'b0;
        else     out_valid <= col_v;
        out_col      <= t_col;
        out_last     <= t_last && col_v;
        out_coef     <= coef_c;
        out_replaced <= t_mask;
        out_approx   <= t_approx;
    end

endmodule

`default_nettype wire
