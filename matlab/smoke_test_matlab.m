function smoke_test_matlab()
%SMOKE_TEST_MATLAB Check the MATLAB/Octave model against the golden vectors.
%   Runs from any folder:   matlab -batch "run('matlab/smoke_test_matlab.m')"
%                           octave --no-gui --eval "cd matlab; smoke_test_matlab"
%   Raises an error (non-zero exit status in batch mode) on any mismatch.

here = fileparts(mfilename('fullpath'));
vec  = fullfile(here, '..', 'vectors');
nerr = 0;

% 1) floating-point Loeffler data flow == orthonormal DCT-II
T = dct_matrix8();
x = 100 * sin((1:800).' * 0.37 + (0:7) * 1.3);        % 800 deterministic test rows
e = max(max(abs(loeffler_dct8(x) / sqrt(8) - x * T.')));
fprintf('float Loeffler vs DCT matrix  : max |error| = %.3g\n', e);
if e > 1e-9, nerr = nerr + 1; end

% 2) quantised constants (CF = 13), shared with RTL and HLS
K = ldct_constants(13);
ref = [4433 6270 15137 6811 -2260 11363 8035 -6436 9633 11585];
got = [K.E_K K.E_A K.E_B K.O3_K K.O3_A K.O3_B K.O1_K K.O1_A K.O1_B K.R2];
fprintf('rotation constants (CF = 13)  : %d mismatches\n', nnz(got ~= ref));
nerr = nerr + nnz(got ~= ref);

% 3) bit-exact 2-D processor on both vector sets
modes = {'exact', 'tsc'};
for i = 1:numel(modes)
    cfg = default_config();
    if strcmp(modes{i}, 'exact')
        cfg.tsc_enable  = false;
        cfg.approx_mode = 0;
    end
    [blocks, Fexp, sexp] = read_vectors(fullfile(vec, modes{i}));
    [F, info] = dct2d_modified_loeffler(blocks, cfg);
    s = double(info.replaced) * (2 .^ (0:7)).' + 256 * double(info.approx);
    bad = nnz(F ~= Fexp) + nnz(s ~= sexp);
    fprintf('golden vectors %-6s          : %d blocks, %d mismatches\n', modes{i}, size(blocks, 3), bad);
    nerr = nerr + bad;
end

if nerr == 0
    fprintf('PASS: MATLAB model matches the golden vectors\n');
else
    error('smoke_test_matlab:fail', 'FAIL: %d mismatches', nerr);
end
end
