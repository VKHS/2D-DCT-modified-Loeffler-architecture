function [C, info] = dct2d_image(img, cfg)
%DCT2D_IMAGE Blockwise 2-D DCT of a grey 8-bit image with the processor model.
%   [C, INFO] = DCT2D_IMAGE(IMG, CFG) returns the coefficient image C (same
%   size as IMG, 8x8 blocks of orthonormal DCT coefficients) and the TSC
%   decisions per block (INFO.replaced, INFO.approx on the block grid).

if nargin < 2, cfg = default_config(); end
img = double(img);
[h, w] = size(img);
H = 8 * ceil(h / 8);
W = 8 * ceil(w / 8);
img = img([1:h, h * ones(1, H - h)], [1:w, w * ones(1, W - w)]);   % edge padding
B = reshape(permute(reshape(img, 8, H / 8, 8, W / 8), [1 3 2 4]), 8, 8, []);
[F, info] = dct2d_modified_loeffler(B, cfg);
C = reshape(permute(reshape(F, 8, 8, H / 8, W / 8), [1 3 2 4]), H, W);
C = C(1:h, 1:w);
info.grid = [H / 8, W / 8];
info.approx_map   = reshape(info.approx, H / 8, W / 8);
info.replaced_map = reshape(sum(info.replaced, 2), H / 8, W / 8);
end
