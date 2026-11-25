// ============================================================================
//  dct2d_loeffler.cpp  -  Vitis HLS implementation
//
//      row_pass : level shift, row engine, TSC unit (stage 1 + stage-2 decision)
//      col_pass : column engine, final normalisation, triangular zone
//
//  The two passes form a DATAFLOW region: the row spectra buffer becomes a
//  ping-pong channel, so consecutive calls overlap and each loop is pipelined
//  with II=1 (one row / one column per clock).
// ============================================================================
#include "dct2d_loeffler.h"

// rotation constants round(c * 2^13)   (python/loeffler2d/constants.py)
static const int E_K = 4433;    // sqrt2*cos(6pi/16)
static const int E_A = 6270;    // sqrt2*(cos(2pi/16)-cos(6pi/16))
static const int E_B = 15137;   // sqrt2*(cos(2pi/16)+cos(6pi/16))
static const int O3_K = 6811;   // cos(3pi/16)
static const int O3_A = -2260;  // cos(5pi/16)-cos(3pi/16)
static const int O3_B = 11363;  // cos(3pi/16)+cos(5pi/16)
static const int O1_K = 8035;   // cos(pi/16)
static const int O1_A = -6436;  // cos(7pi/16)-cos(pi/16)
static const int O1_B = 9633;   // cos(pi/16)+cos(7pi/16)
static const int R2 = 11585;    // sqrt2

// SAD thresholds TINT[m] = round(m * 0.001 * 255 * 8 * sqrt8 * 2^RF), RF = 2
static const uint16_t TINT[16] = {0,   23,  46,  69,  92,  115, 138, 162,
                                  185, 208, 231, 254, 277, 300, 323, 346};

static inline acc_t rnd_shift(acc_t v) {
#pragma HLS INLINE
    return (v + (acc_t)(1 << (DCT_CF - 1))) >> DCT_CF;
}

// ---------------------------------------------------------------------------
//  8-point modified Loeffler engine: 11 multiplications, 29 additions.
//  Output scale sqrt(8); the LSB weight of the input is preserved.
// ---------------------------------------------------------------------------
template <typename T>
static void dct8_loeffler(const T x[8], T y[8]) {
#pragma HLS INLINE
    // stage 1: butterflies
    T a0 = x[0] + x[7], a7 = x[0] - x[7];
    T a1 = x[1] + x[6], a6 = x[1] - x[6];
    T a2 = x[2] + x[5], a5 = x[2] - x[5];
    T a3 = x[3] + x[4], a4 = x[3] - x[4];

    // stage 2: even butterflies, odd rotations (3pi/16 and pi/16)
    T b0 = a0 + a3, b3 = a0 - a3;
    T b1 = a1 + a2, b2 = a1 - a2;
    T s47 = a4 + a7, s56 = a5 + a6;
    acc_t p_o3k = (acc_t)O3_K * s47;
    acc_t p_o1k = (acc_t)O1_K * s56;
    T b4 = rnd_shift(p_o3k + (acc_t)O3_A * a7);
    T b7 = rnd_shift(p_o3k - (acc_t)O3_B * a4);
    T b5 = rnd_shift(p_o1k + (acc_t)O1_A * a6);
    T b6 = rnd_shift(p_o1k - (acc_t)O1_B * a5);

    // stage 3: even DC/Nyquist and sqrt2-rotation 6pi/16, odd butterflies
    T y0 = b0 + b1, y4 = b0 - b1;
    T s23 = b2 + b3;
    acc_t p_ek = (acc_t)E_K * s23;
    T y2 = rnd_shift(p_ek + (acc_t)E_A * b3);
    T y6 = rnd_shift(p_ek - (acc_t)E_B * b2);
    T d4 = b4 + b6, d6 = b4 - b6;
    T d7 = b7 + b5, d5 = b7 - b5;

    // stage 4: odd outputs
    y[0] = y0;
    y[1] = d7 + d4;
    y[2] = y2;
    y[3] = rnd_shift((acc_t)R2 * d5);
    y[4] = y4;
    y[5] = rnd_shift((acc_t)R2 * d6);
    y[6] = y6;
    y[7] = d7 - d4;
}

// ---------------------------------------------------------------------------
//  Row pass with the Threshold Set and Comparison unit
// ---------------------------------------------------------------------------
static void row_pass(const pixel_t in[8][8], row_t rows[8][8], const tsc_config_t &cfg,
                     block_status_t &status, bool &approx, uint8_t &zone) {
    row_t ref[8];
#pragma HLS ARRAY_PARTITION variable=ref complete
    uint8_t tidx = cfg.t0;
    uint8_t nsim = 0;
    uint8_t mask = 0;

ROW_LOOP:
    for (int r = 0; r < 8; r++) {
#pragma HLS PIPELINE II=1
        row_t x[8], y[8];
#pragma HLS ARRAY_PARTITION variable=x complete
#pragma HLS ARRAY_PARTITION variable=y complete
        for (int k = 0; k < 8; k++) x[k] = (row_t)(((int)in[r][k] - 128) * (1 << DCT_RF));
        dct8_loeffler<row_t>(x, y);

        bool similar = false;
        if (r > 0 && cfg.enable) {
            sad_t sad = 0;
            for (int k = 0; k < 8; k++) {
                int d = (int)y[k] - (int)ref[k];
                sad += (sad_t)(d < 0 ? -d : d);
            }
            similar = sad < (sad_t)TINT[tidx];
        }
        for (int k = 0; k < 8; k++) {
            if (!similar) ref[k] = y[k];
            rows[r][k] = ref[k];
        }
        if (r > 0 && !similar) {
            int t = tidx + cfg.step;
            tidx = (uint8_t)(t > cfg.tmax ? cfg.tmax : t);
        }
        if (similar) {
            mask |= (uint8_t)(1u << r);
            if (r <= 3) nsim++;
        }
    }
    approx = (cfg.approx_mode == 2) || (cfg.approx_mode == 1 && cfg.enable && nsim >= cfg.min_similar);
    status.replaced = mask;
    status.approx = approx;
    zone = cfg.zone;
}

// ---------------------------------------------------------------------------
//  Column pass, final normalisation (scale 8 * 2^RF) and triangular zone
// ---------------------------------------------------------------------------
static void col_pass(const row_t rows[8][8], coef_t out[8][8], bool approx, uint8_t zone) {
COL_LOOP:
    for (int u = 0; u < 8; u++) {
#pragma HLS PIPELINE II=1
        col_t x[8], z[8];
#pragma HLS ARRAY_PARTITION variable=x complete
#pragma HLS ARRAY_PARTITION variable=z complete
        for (int r = 0; r < 8; r++) x[r] = (col_t)rows[r][u];
        dct8_loeffler<col_t>(x, z);
        for (int v = 0; v < 8; v++) {
            int f = ((int)z[v] + (1 << (DCT_RF + 2))) >> (DCT_RF + 3);
            out[v][u] = (approx && (u + v) > zone) ? (coef_t)0 : (coef_t)f;
        }
    }
}

// ---------------------------------------------------------------------------
void dct2d_modified_loeffler(const pixel_t in[8][8], coef_t out[8][8],
                             const tsc_config_t &cfg, block_status_t &status) {
#pragma HLS INTERFACE ap_memory port=in
#pragma HLS INTERFACE ap_memory port=out
#pragma HLS ARRAY_PARTITION variable=in complete dim=2
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS DATAFLOW
    row_t rows[8][8];
#pragma HLS ARRAY_PARTITION variable=rows complete dim=0
    bool approx;
    uint8_t zone;
    row_pass(in, rows, cfg, status, approx, zone);
    col_pass(rows, out, approx, zone);
}
