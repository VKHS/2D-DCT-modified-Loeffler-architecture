# Verification

All four implementations are checked against one set of golden vectors produced by the
bit-accurate Python model.

```
python/loeffler2d (golden model)
        │  python/scripts/gen_vectors.py
        ▼
vectors/exact   vectors/tsc   vectors/engine
   │                │              │
   ├── rtl/tb/tb_dct2d_top.v      (Icarus Verilog / xsim)       ├── rtl/tb/tb_loeffler_dct8.v
   ├── hls/tb_dct2d_loeffler.cpp  (g++ C-sim, Vitis csim/cosim)
   ├── matlab/smoke_test_matlab.m (MATLAB / Octave)
   └── python/tests               (pytest)
```

## Vector sets

| set | blocks | configuration | purpose |
|---|---|---|---|
| `exact` | 63 | TSC off, approximation off | bit-exact fixed-point 2-D DCT |
| `tsc` | 63 | TSC on: t0 = 1, step = 1, tmax = 10, min_sim = 3, mode = TSC, zone = 5 | row replacement, threshold adaptation, stage 2 |
| `engine` | 512 vectors | row-engine input range, 16 extreme sign patterns first | stand-alone 1-D engine |

The 63 blocks (`python/loeffler2d/images.py::vector_blocks`) cover: all-0, all-255 and
mid-grey blocks; 16 worst-case sign patterns of the DCT basis (largest coefficient
magnitudes, extreme word-length corners); horizontal, vertical and diagonal ramps; step
edges; two checkerboards; a block with eight identical rows (every row replaced); a block
with ±1 noise on identical rows; a block whose first four rows are identical (stage 2
only); 24 uniform-noise blocks; 8 crops of a synthetic natural scene. In `tsc` mode 149 of
the 441 comparable rows are replaced and 15 blocks are approximated.

File formats are described in [`python/loeffler2d/vectors.py`](../python/loeffler2d/vectors.py):
one hex word per line, lane 0 in the least significant bits.

## Test benches

| test bench | checks |
|---|---|
| `rtl/tb/tb_loeffler_dct8.v` | 512 engine vectors back to back; prints the measured latency (7) |
| `rtl/tb/tb_dct2d_top.v` | every coefficient, column order, `out_last`, and `{out_approx, out_replaced}` of every block; `GAPS=1` inserts random idle cycles between input rows |
| `hls/tb_dct2d_loeffler.cpp` | both vector sets through the HLS top function; return code for `csim_design` / `cosim_design` |
| `matlab/smoke_test_matlab.m` | float engine vs DCT matrix, quantised constants, both vector sets |
| `python/tests/test_models.py` | float/fixed engines, 11 mult / 29 add, constant widths, word-length bounds, 12-bit output range, threshold table, TSC behaviour, operation counting, committed vectors |

## Running

```bash
bash scripts/smoke_test.sh          # everything that is installed
bash scripts/run_rtl_sim.sh         # Icarus Verilog only (5 runs)
bash scripts/run_xsim.sh            # Vivado simulator
make hls-csim                       # HLS C simulation with g++
vitis_hls -f hls/run_hls.tcl        # Vitis HLS csim + csynth (DCT_COSIM=1 adds co-simulation)
```

The GitHub workflow [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) installs Icarus
Verilog and Octave and runs `scripts/smoke_test.sh` on every push.

## Regenerating vectors

The model and the vectors are deterministic. After a change of word lengths or TSC
parameters:

```bash
python3 python/scripts/gen_vectors.py          # rewrite vectors/
python3 python/scripts/gen_rtl_constants.py --cf 13 --rf 2   # Verilog/HLS constants, TINT table
```
