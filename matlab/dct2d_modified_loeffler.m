function [F, info] = dct2d_modified_loeffler(blocks, cfg)
%DCT2D_MODIFIED_LOEFFLER Bit-accurate model of the 2-D DCT processor.
%   [F, INFO] = DCT2D_MODIFIED_LOEFFLER(BLOCKS, CFG) transforms the 8x8xN
%   stack of 8-bit pixel blocks (BLOCKS(r+1, k+1, n) = pixel k of row r):
%
%       x = (p - 128) * 2^RF            level shift, RF guard bits
%       Y = engine(x)    per row        scale sqrt(8)
%       Y = TSC(Y)                      stage 1: row replacement
%       Z = engine(Y.')  per column     scale 8
%       F = floor((Z + 2^(RF+2)) / 2^(RF+3))
%       F(u+v > zone) = 0  for blocks selected by stage 2
%
%   F(v+1, u+1, n) is the orthonormal 2-D DCT coefficient of vertical
%   frequency v and horizontal frequency u, identical to rtl/dct2d_top.v.
%   INFO holds the TSC decisions (replaced, approx, sad, tidx, rows).

if nargin < 2, cfg = default_config(); end
n = size(blocks, 3);
P = double(blocks);

% ------------------------------------------------------------ row pass
X  = (P - 128) * 2^cfg.rf;
Xr = reshape(permute(X, [2 1 3]), 8, []).';          % (8N)-by-8, one row per line
Yr = loeffler_dct8_fixed(Xr, cfg.cf);
Y  = permute(reshape(Yr.', 8, 8, n), [2 1 3]);        % (r, u, n)

% ------------------------------------------------------------ TSC unit
[R, replaced, approx, sad, tidx] = tsc_unit(Y, cfg);

% ------------------------------------------------------------ column pass
Rc = reshape(R, 8, []).';                             % (8N)-by-8, one column per line
Zc = loeffler_dct8_fixed(Rc, cfg.cf);
Z  = reshape(Zc.', 8, 8, n);                          % (v, u, n)

s = cfg.rf + 3;
F = floor((Z + 2^(s - 1)) / 2^s);

% ------------------------------------------- stage 2: triangular zone
mask = triangle_mask(cfg.zone);
for b = reshape(find(approx), 1, [])
    Fb = F(:, :, b);
    Fb(~mask) = 0;
    F(:, :, b) = Fb;
end

info = struct('replaced', replaced, 'approx', approx, 'sad', sad, 'tidx', tidx);
info.rows = R;
end
