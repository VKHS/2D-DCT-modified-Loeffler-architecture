function p = psnr_u8(ref, test)
%PSNR_U8 Peak signal-to-noise ratio of two 8-bit images in dB.
e = mean((double(ref(:)) - double(test(:))).^2);
if e == 0, p = Inf; else, p = 10 * log10(255^2 / e); end
end
