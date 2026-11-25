# Architecture reference

This page documents the RTL in [`rtl/`](../rtl) module by module: data flow, pipeline,
word lengths, the Threshold Set and Comparison unit, the in-place transpose buffer, the
controller and the interface timing. The Python model
[`python/loeffler2d`](../python/loeffler2d), the MATLAB model [`matlab/`](../matlab) and the HLS
code [`hls/`](../hls) compute exactly the same integers.

## 1. Module hierarchy

```
dct2d_top                    rtl/dct2d_top.v
├── u_row_engine             loeffler_dct8      WI = 10, WD = 13, TAGW = 3   (row index)
├── u_tsc                    tsc_unit           W = 13
├── u_ctrl                   dct2d_controller
├── u_transpose              transpose_buffer   W = 13
└── u_col_engine             loeffler_dct8      WI = 13, WD = 16, TAGW = 13  (approx, last, col, mask)
```

| block | function | latency |
|---|---|---|
| level shift | `x = (p − 128) << RF`: MSB inversion and RF = 2 zero LSBs | 0 (wires) |
| row engine | 8-point modified Loeffler DCT of one row per clock, scale √8 | 7 clk |
| TSC unit | compare row r with the spectrum kept for row r−1, replace, adapt threshold, stage-2 flag | 2 clk |
| transpose buffer | write one row per clock, read one column per clock, in place | write at the clock edge, read combinational |
| controller | row tags, mask/flag latching, column read FSM | — |
| column engine | 8-point modified Loeffler DCT of one column per clock, scale 8 in total | 7 clk |
| normalise | `F = (Z + 2^(RF+2)) >>> (RF+3)`, zone mask `u + v ≤ Z` when stage 2 is active | 1 clk |

Throughput: one 8×8 block every 8 clocks. Latency: row 7 of a block enters at clock *t*,
column 0 of that block leaves at clock *t* + 18 (7 row engine + 2 TSC + 1 buffer + 7 column
engine + 1 output register); the RTL waveform in
[`docs/figures/rtl_waveform.png`](figures/rtl_waveform.png) shows exactly this.

## 2. Loeffler engine (`loeffler_dct8.v`)

The engine computes `Y = √8 · C8 · x` with the four stages of Loeffler, Ligtenberg and
Moschytz (ICASSP 1989), 11 multiplications and 29 additions:

| stage | even part (lanes 0..3) | odd part (lanes 4..7) | add | mult |
|---|---|---|---|---|
| 1 | `a_i = x_i + x_{7−i}` | `a_{7−i} = x_i − x_{7−i}` | 8 | 0 |
| 2 | `b0 = a0 + a3`, `b3 = a0 − a3`, `b1 = a1 + a2`, `b2 = a1 − a2` | `(b4, b7) = rot_{3π/16}(a4, a7)`, `(b5, b6) = rot_{π/16}(a5, a6)` | 10 | 6 |
| 3 | `Y0 = b0 + b1`, `Y4 = b0 − b1`, `(Y2, Y6) = √2·rot_{6π/16}(b2, b3)` | `d4 = b4 + b6`, `d6 = b4 − b6`, `d7 = b7 + b5`, `d5 = b7 − b5` | 9 | 3 |
| 4 | — | `Y1 = d7 + d4`, `Y7 = d7 − d4`, `Y3 = √2·d5`, `Y5 = √2·d6` | 2 | 2 |

Each rotation uses the 3-multiplier form `z = K(xp + xq)`, `yp = z + A·xq`, `yq = z − B·xp`.
The three products are kept at full precision, and every rotation output is rounded once:
`yp = (K(xp + xq) + A·xq + 2^12) >>> 13`. The shared product has the shape of a DSP slice with
its pre-adder (A/D → pre-adder → M register), and each output adds its own product in a
post-adder (P register), so a rotation fits three DSP slices.

| rotation | inputs | K | A | B | constants, CF = 13 |
|---|---|---|---|---|---|
| 3π/16 | (a4, a7) | cos 3π/16 | cos 5π/16 − cos 3π/16 | cos 3π/16 + cos 5π/16 | 6811, −2260, 11363 |
| π/16 | (a5, a6) | cos π/16 | cos 7π/16 − cos π/16 | cos π/16 + cos 7π/16 | 8035, −6436, 9633 |
| √2 · 6π/16 | (b2, b3) | √2 cos 6π/16 | √2 (cos 2π/16 − cos 6π/16) | √2 (cos 2π/16 + cos 6π/16) | 4433, 6270, 15137 |
| √2 | d5, d6 | √2 | | | 11585 |

### Pipeline

| register | content |
|---|---|
| P1 | stage-1 butterflies `a0..a7` |
| P2 | even butterflies `b0..b3`, pre-adds `s47 = a4 + a7`, `s56 = a5 + a6`, odd inputs delayed |
| P3 | six odd products, `Y0`, `Y4`, pre-add `s23 = b2 + b3` |
| P4 | odd post-adds with rounding `b4..b7`, three even products |
| P5 | stage-3 odd butterflies `d4..d7`, even `Y2`, `Y6` — even half complete |
| P6 | `Y1`, `Y7`, two √2 products |
| P7 | rounding of `Y3`, `Y5`, aligned output register — odd half complete |

A `valid` bit and a `TAGW`-bit side band travel with the data through a 7-stage shift
register, so the row engine carries the row index and the column engine carries
`{approx, last, column, mask}`.

### Word lengths

`python/loeffler2d/fixed_point.py::engine_ranges` pushes, for every node, the input vector
that maximises (and minimises) that node through the bit-exact engine. With 8-bit pixels,
RF = 2 and CF = 13:

| engine | input range | widest node | data width WD | product width |
|---|---|---|---|---|
| row | −512 … 508 (10 bit) | 13 bit (Y0 = −4096 … 4064) | 13 | 28 |
| column | −4096 … 4080 (13 bit) | 16 bit (Y0 = −32768 … 32640) | 16 | 31 |

The output `F = round(Z / 32)` lies in −1024 … 1020 and is delivered as 12-bit two's
complement. Against the floating-point orthonormal DCT the maximum error is 0.71 LSB over
6000 random blocks and the blocks of *camera*, where 3.6 % of the coefficients differ from
ideal rounding by one ([`fixed_point_accuracy.png`](figures/fixed_point_accuracy.png)); on the
eight sample images it is 0.69–0.76 LSB.

## 3. TSC unit (`tsc_unit.v`)

### Algorithm (per 8×8 block)

```
m   <- t0                          # threshold index, 0.001 per step
ref <- Y[0]                        # spectrum kept for the previous row
for r = 1 .. 7:
    SAD <- sum_k |Y[r][k] - ref[k]|
    if enable and SAD < TINT[m]:   # stage 1: rows r-1 and r are similar
        Y'[r] <- ref               #          replace row r by its predecessor
    else:
        Y'[r] <- Y[r];  ref <- Y[r]
        m <- min(m + step, tmax)   # adaptive: raise the threshold for the next row
approx <- (mode == always) or (mode == TSC and enable and
          similar pairs among (1,0), (2,1), (3,2) >= min_sim)      # stage 2
```

`TINT[m] = round(m · 0.001 · 255 · 8 · √8 · 2^RF)` converts the threshold into the units of
the Loeffler-scaled row words, so `SAD < TINT[m]` is the same test as
`mean_k |X_r[k] − X_ref[k]| / 255 < 0.001·m` on the orthonormal row DCT X of 8-bit pixels.
For RF = 2 the table is `0, 23, 46, 69, 92, 115, 138, 162, 185, 208, 231, 254, 277, 300, 323, 346`.

### Hardware

* **Stage A** computes two sums of absolute differences in parallel:
  `sad_a = SAD(Y_r, Y_{r−1})` against the previous input row, and
  `sad_b = SAD(Y_r, ref)` against the kept spectrum. Each is 8 subtractors, 8 absolute
  values and a 7-adder tree (17-bit result).
* **Stage B** selects `sad_b` when row r−1 was replaced and `sad_a` otherwise; both
  selections equal `SAD(Y_r, ref_{r−1})`, with or without idle cycles between rows. The
  loop that remains is `last_rep → mux → comparator → last_rep`, short enough for one row
  per clock.
* `m` is a 4-bit register, reset to `t0` on row 0; `TINT` is a 16 × 10-bit table.
* The stage-2 flag is final when row 3 leaves the unit; the controller latches it with
  the block.

## 4. Transpose buffer (`transpose_buffer.v`)

A single 64-word register file. Block k is written in orientation o and read in the same
orientation while block k+1 is written in orientation ¬o:

* orientation 0: element (r, u) at `mem[8r + u]` — row r occupies memory row r;
* orientation 1: element (r, u) at `mem[8u + r]` — row r occupies memory column r.

Reading column c of block k touches exactly the eight cells that row c of block k+1 is
written to. Column c of block k is read on the c-th clock after block k completes, and
row c of block k+1 cannot arrive earlier, so a cell is never overwritten before it is read
(read-before-write in the same clock). The buffer therefore needs 64 words instead of the
128 of a ping-pong pair ([`transpose_buffer.png`](figures/transpose_buffer.png)).

## 5. Controller (`dct2d_controller.v`)

* `in_row` counts valid input rows modulo 8 and tags them (blocks arrive row by row).
* `acc_mask[r]` collects the replacement decisions of the block being written.
* Read FSM `S_IDLE → S_READ`: the clock after row 7 of a block is written, columns 0..7 are
  issued on eight consecutive clocks together with the block's mask and stage-2 flag.
  When the next block completes in the clock that issues column 7, the FSM restarts at
  column 0 without a gap ([`controller_fsm.png`](figures/controller_fsm.png)).

## 6. Interface of `dct2d_top`

| port | dir | width | description |
|---|---|---|---|
| `clk`, `rst` | in | 1 | clock, synchronous active-high reset |
| `cfg_tsc_en` | in | 1 | stage 1 on/off |
| `cfg_t0`, `cfg_step`, `cfg_tmax` | in | 4 each | threshold index: default, increment, maximum (× 0.001) |
| `cfg_min_sim` | in | 2 | similar pairs among rows 0..3 that trigger stage 2 |
| `cfg_approx_mode` | in | 2 | 0 off, 1 TSC-triggered, 2 every block |
| `cfg_zone` | in | 4 | keep coefficients with `u + v ≤ cfg_zone` in approximated blocks |
| `in_valid` | in | 1 | a pixel row is present |
| `in_row` | in | 64 | pixel k of the row in bits `[8k +: 8]`; rows 0..7 of a block in order |
| `out_valid` | out | 1 | a coefficient column is present |
| `out_col` | out | 3 | horizontal frequency u of the column |
| `out_last` | out | 1 | column 7 of a block |
| `out_coef` | out | 96 | `F[v][u]` in bits `[12v +: 12]`, two's complement |
| `out_replaced` | out | 8 | replacement mask of the block (valid with `out_last`) |
| `out_approx` | out | 1 | stage-2 flag of the block (valid with `out_last`) |

Input rows may be separated by idle cycles; outputs follow as soon as a block is complete.
Configuration inputs are quasi-static and should change only between frames.
