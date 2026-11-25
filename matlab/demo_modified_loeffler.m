%DEMO_MODIFIED_LOEFFLER Exact vs. operation-reduced 2-D DCT on a test image.
%   Runs the bit-accurate processor model twice (TSC off / on), reconstructs
%   the image with the inverse DCT, applies JPEG-style quantisation and plots
%   the decisions of the Threshold Set and Comparison unit per block.

clear; close all;
try
    img = imread('cameraman.tif');                 % Image Processing Toolbox
catch
    [x, y] = meshgrid(linspace(0, 1, 256));        % synthetic fallback scene
    img = 120 + 60 * y + 40 * sin(18 * x) .* (x > 0.55) + 50 * ((x - 0.3).^2 + (y - 0.4).^2 < 0.02);
    img = uint8(min(max(img, 0), 255));
end
if size(img, 3) == 3
    img = uint8(0.299 * double(img(:, :, 1)) + 0.587 * double(img(:, :, 2)) + 0.114 * double(img(:, :, 3)));
end

cfg_exact = default_config();
cfg_exact.tsc_enable  = false;
cfg_exact.approx_mode = 0;
cfg_tsc = default_config();                         % t0 = 0.001, step 0.001, max 0.010

[C0, ~]  = dct2d_image(img, cfg_exact);
[C1, i1] = dct2d_image(img, cfg_tsc);
r0 = idct2d_image(C0);
r1 = idct2d_image(C1);

fprintf('exact fixed-point  : PSNR %.2f dB\n', psnr_u8(img, r0));
fprintf('TSC + column approx: PSNR %.2f dB, rows replaced %.1f %%, blocks approximated %.1f %%\n', ...
        psnr_u8(img, r1), 100 * mean(mean(i1.replaced(:, 2:8))), 100 * mean(i1.approx));

% JPEG-style quantisation (quality 75)
Q  = jpeg_qtable(75);
[h, w] = size(img);
Qi = repmat(Q, ceil(h / 8), ceil(w / 8));
Qi = Qi(1:h, 1:w);
j0 = idct2d_image(round(C0 ./ Qi) .* Qi);
j1 = idct2d_image(round(C1 ./ Qi) .* Qi);
fprintf('JPEG q75           : exact %.2f dB, TSC %.2f dB\n', psnr_u8(img, j0), psnr_u8(img, j1));

figure('Name', 'Modified Loeffler 2-D DCT', 'Color', 'w');
subplot(2, 3, 1); imagesc(img); axis image off; title('input');
subplot(2, 3, 2); imagesc(log10(1 + abs(C1))); axis image off; title('log_{10}(1+|F|), 8x8 blocks');
subplot(2, 3, 3); imagesc(r1); axis image off; title(sprintf('TSC reconstruction %.1f dB', psnr_u8(img, r1)));
subplot(2, 3, 4); imagesc(i1.replaced_map, [0 7]); axis image off; colorbar; title('rows replaced per block');
subplot(2, 3, 5); imagesc(i1.approx_map, [0 1]); axis image off; title('column-wise approximation');
subplot(2, 3, 6); imagesc(abs(double(r1) - double(r0)), [0 8]); axis image off; colorbar; title('|TSC - exact|');
colormap(gray);
