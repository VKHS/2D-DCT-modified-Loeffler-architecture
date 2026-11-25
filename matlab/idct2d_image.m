function img = idct2d_image(C)
%IDCT2D_IMAGE Floating-point blockwise inverse DCT, +128, rounded to uint8.
[h, w] = size(C);
H = 8 * ceil(h / 8);
W = 8 * ceil(w / 8);
Cp = zeros(H, W);
Cp(1:h, 1:w) = double(C);
T = dct_matrix8();
B = reshape(permute(reshape(Cp, 8, H / 8, 8, W / 8), [1 3 2 4]), 8, 8, []);
for b = 1:size(B, 3)
    B(:, :, b) = T.' * B(:, :, b) * T;
end
img = reshape(permute(reshape(B, 8, 8, H / 8, W / 8), [1 3 2 4]), H, W);
img = uint8(min(max(round(img(1:h, 1:w) + 128), 0), 255));
end
