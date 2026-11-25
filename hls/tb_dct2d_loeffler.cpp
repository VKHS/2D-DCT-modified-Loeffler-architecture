// ============================================================================
//  tb_dct2d_loeffler.cpp  -  C simulation / co-simulation test bench
//
//      ./hls_csim [vector_dir]          (default: vectors)
//
//  Runs both golden-vector sets (exact, tsc) through the HLS top function and
//  compares every coefficient and the per-block TSC status.  Returns 0 on
//  success, as required by csim_design / cosim_design.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "dct2d_loeffler.h"

static std::vector<std::string> read_lines(const std::string &path) {
    std::vector<std::string> out;
    std::ifstream f(path.c_str());
    std::string s;
    while (f >> s) out.push_back(s);
    return out;
}

static int hexval(const std::string &s, size_t pos, size_t n) {
    return (int)std::strtol(s.substr(pos, n).c_str(), nullptr, 16);
}

static int run_mode(const std::string &dir, const char *mode, const tsc_config_t &cfg) {
    std::string base = dir + "/" + mode + "/";
    std::vector<std::string> rows = read_lines(base + "input_rows.hex");
    std::vector<std::string> cols = read_lines(base + "expected_cols.hex");
    std::vector<std::string> stat = read_lines(base + "expected_status.hex");
    if (rows.empty() || rows.size() != cols.size() || rows.size() != 8 * stat.size()) {
        std::printf("FAIL: cannot read vectors in %s\n", base.c_str());
        return 1;
    }
    const size_t nb = stat.size();
    int errors = 0;
    for (size_t b = 0; b < nb; b++) {
        pixel_t in[8][8];
        coef_t out[8][8];
        block_status_t st;
        for (int r = 0; r < 8; r++) {
            const std::string &w = rows[8 * b + r];  // 16 hex digits, pixel 0 last
            for (int k = 0; k < 8; k++) in[r][k] = (pixel_t)hexval(w, 14 - 2 * k, 2);
        }
        dct2d_modified_loeffler(in, out, cfg, st);
        for (int u = 0; u < 8; u++) {
            const std::string &w = cols[8 * b + u];  // 24 hex digits, F[0][u] last
            for (int v = 0; v < 8; v++) {
                int e = hexval(w, 21 - 3 * v, 3);
                if (e >= 2048) e -= 4096;
                if ((int)out[v][u] != e) {
                    if (++errors <= 10)
                        std::printf("MISMATCH %s block %zu F[%d][%d]: got %d expected %d\n", mode, b, v, u,
                                    (int)out[v][u], e);
                }
            }
        }
        int es = hexval(stat[b], 0, 3);
        int gs = st.replaced | (st.approx ? 256 : 0);
        if (gs != es) {
            if (++errors <= 10) std::printf("STATUS MISMATCH %s block %zu: got %03x expected %03x\n", mode, b, gs, es);
        }
    }
    if (errors == 0)
        std::printf("PASS: HLS dct2d_modified_loeffler mode=%s blocks=%zu\n", mode, nb);
    else
        std::printf("FAIL: HLS mode=%s errors=%d\n", mode, errors);
    return errors ? 1 : 0;
}

int main(int argc, char **argv) {
    std::string dir = argc > 1 ? argv[1] : "vectors";
    tsc_config_t exact = {false, 1, 1, 10, 3, 0, 5};
    tsc_config_t tsc = {true, 1, 1, 10, 3, 1, 5};
    int fail = run_mode(dir, "exact", exact);
    fail |= run_mode(dir, "tsc", tsc);
    std::printf(fail ? "HLS C-SIM FAILED\n" : "HLS C-SIM PASSED\n");
    return fail;
}
