function T = dct_matrix8()
%DCT_MATRIX8 Orthonormal 8-point DCT-II matrix, X = T*x, F = T*B*T.'
[k, m] = ndgrid(0:7, 0:7);
T = cos((2 * m + 1) .* k * pi / 16);
T(1, :)     = T(1, :) * sqrt(1 / 8);
T(2:end, :) = T(2:end, :) * sqrt(2 / 8);
end
