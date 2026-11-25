# MATLAB / Octave model

Bit-accurate model of the processor, equal to `python/loeffler2d`, `rtl/` and `hls/`.

| file | content |
|---|---|
| `loeffler_dct8.m` | floating-point 8-point Loeffler data flow (4 stages, 11 mult, 29 add), returns every stage |
| `loeffler_dct8_fixed.m` | integer engine: constants `round(c*2^13)`, fused rotations, one rounding per output |
| `tsc_unit.m` | Threshold Set and Comparison unit: row replacement, adaptive threshold, stage-2 decision |
| `dct2d_modified_loeffler.m` | complete 2-D processor for an 8x8xN stack of blocks |
| `dct2d_image.m`, `idct2d_image.m` | image wrappers (edge padding, blockwise inverse) |
| `default_config.m` | word lengths and TSC parameters |
| `ldct_constants.m`, `threshold_table.m`, `triangle_mask.m`, `dct_matrix8.m` | constants and tables |
| `read_vectors.m`, `read_hex_lines.m` | golden-vector reader |
| `smoke_test_matlab.m` | checks the model against `vectors/exact` and `vectors/tsc` |
| `demo_modified_loeffler.m` | exact vs. TSC on an image, JPEG-style quantisation, decision maps |

```matlab
cd matlab
smoke_test_matlab            % PASS: MATLAB model matches the golden vectors
demo_modified_loeffler       % figures + PSNR report
```

Octave: `octave --no-gui --eval "cd matlab; smoke_test_matlab"`. No toolbox is required;
the demo uses `cameraman.tif` when the Image Processing Toolbox is installed and a synthetic
scene otherwise.
