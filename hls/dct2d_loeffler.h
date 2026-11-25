// ============================================================================
//  dct2d_loeffler.h  -  HLS model of the modified-Loeffler 2-D DCT processor
//
//  Bit-exact with rtl/dct2d_top.v and python/loeffler2d (same constants,
//  rounding and word lengths).  With Vitis HLS the arbitrary-precision types
//  of <ap_int.h> are used; a plain C++17 compiler falls back to standard
//  integers, which give identical results because no node ever overflows its
//  declared width.
// ============================================================================
#ifndef DCT2D_LOEFFLER_H
#define DCT2D_LOEFFLER_H

#include <stdint.h>

// ---------------------------------------------------------------- word lengths
#define DCT_CF 13  // fractional bits of the rotation constants
#define DCT_RF 2   // guard bits carried from the row to the column pass
#define DCT_WR 13  // row-engine data width   (8 + RF + 3)
#define DCT_WK 16  // column-engine data width (WR + 3)
#define DCT_WO 12  // output coefficient width

#if defined(__has_include)
#if __has_include(<ap_int.h>)
#define DCT_HAVE_AP_INT 1
#endif
#endif

#ifdef DCT_HAVE_AP_INT
#include <ap_int.h>
typedef ap_uint<8> pixel_t;
typedef ap_int<DCT_WR> row_t;           // row spectra (Loeffler scale sqrt8)
typedef ap_int<DCT_WK> col_t;           // column-engine words (scale 8)
typedef ap_int<DCT_WO> coef_t;          // orthonormal 2-D DCT coefficients
typedef ap_int<DCT_WK + 17> acc_t;      // products and fused rotation sums
typedef ap_uint<DCT_WR + 4> sad_t;      // sum of absolute differences
#else
typedef uint8_t pixel_t;
typedef int32_t row_t;
typedef int32_t col_t;
typedef int16_t coef_t;
typedef int64_t acc_t;
typedef uint32_t sad_t;
#endif

// ------------------------------------------------------------ configuration
struct tsc_config_t {
    bool enable;          // stage 1: row replacement
    uint8_t t0;           // default threshold index        (x 0.001)
    uint8_t step;         // adaptive increment              (x 0.001)
    uint8_t tmax;         // maximum threshold index         (x 0.001)
    uint8_t min_similar;  // similar pairs among rows 0..3 for stage 2 (0..3)
    uint8_t approx_mode;  // 0 off, 1 TSC-triggered, 2 always
    uint8_t zone;         // triangular zone: keep u + v <= zone
};

struct block_status_t {
    uint8_t replaced;  // bit r = row r replaced by row r-1
    bool approx;       // column-wise approximation applied
};

// ------------------------------------------------------------------- top
//  in[r][k]  : pixel k of row r
//  out[v][u] : coefficient of vertical frequency v, horizontal frequency u
void dct2d_modified_loeffler(const pixel_t in[8][8], coef_t out[8][8],
                             const tsc_config_t &cfg, block_status_t &status);

#endif
