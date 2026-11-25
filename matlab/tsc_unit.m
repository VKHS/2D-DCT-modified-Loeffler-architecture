function [R, replaced, approx, sad, tidx] = tsc_unit(Y, cfg)
%TSC_UNIT Threshold Set and Comparison unit (stage 1) and stage-2 decision.
%   [R, REPLACED, APPROX, SAD, TIDX] = TSC_UNIT(Y, CFG) processes the integer
%   row spectra Y(r+1, u+1, n) of N blocks (Loeffler scale, 2^-RF LSB).
%
%   Stage 1: the spectrum of row r is compared with the one kept for row r-1;
%            SAD < TINT(m) -> row r is replaced by its predecessor.  After
%            every dissimilar pair the threshold index m grows by CFG.STEP,
%            saturating at CFG.TMAX (m restarts at CFG.T0 in every block).
%   Stage 2: APPROX(n) is set when at least CFG.MIN_SIMILAR of the pairs
%            (1,0), (2,1), (3,2) were similar (or always, CFG.APPROX_MODE = 2).
%
%   R        row spectra after replacement            8-by-8-by-N
%   REPLACED REPLACED(n, r+1) row r replaced           N-by-8 logical
%   APPROX   column-wise approximation of block n      N-by-1 logical
%   SAD      SAD of row r against its reference        N-by-8
%   TIDX     threshold index applied to row r          N-by-8

n    = size(Y, 3);
tint = threshold_table(cfg.rf, cfg.unit);
R        = Y;
replaced = false(n, 8);
sad      = zeros(n, 8);
tidx     = zeros(n, 8);
m        = cfg.t0 * ones(n, 1);
ref      = reshape(Y(1, :, :), 8, n);      % spectrum kept for the previous row
nsim     = zeros(n, 1);
tidx(:, 1) = m;

for r = 2:8                                % rows 1..7
    tidx(:, r) = m;
    yr  = reshape(Y(r, :, :), 8, n);
    s   = sum(abs(yr - ref), 1).';
    thr = reshape(tint(m + 1), [], 1);
    sim = logical(cfg.tsc_enable) & (s < thr);
    sad(:, r)      = s;
    replaced(:, r) = sim;
    out = yr;
    out(:, sim) = ref(:, sim);
    R(r, :, :) = reshape(out, 1, 8, n);
    ref(:, ~sim) = yr(:, ~sim);
    m(~sim) = min(m(~sim) + cfg.step, cfg.tmax);
    if r <= 4                              % pairs (1,0), (2,1), (3,2)
        nsim = nsim + double(sim);
    end
end

if cfg.approx_mode == 2
    approx = true(n, 1);
elseif cfg.approx_mode == 1 && cfg.tsc_enable
    approx = nsim >= cfg.min_similar;
else
    approx = false(n, 1);
end
end
