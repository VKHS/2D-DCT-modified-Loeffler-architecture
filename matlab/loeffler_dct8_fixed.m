function Y = loeffler_dct8_fixed(X, cf)
%LOEFFLER_DCT8_FIXED Bit-accurate integer 8-point Loeffler DCT (rows of X).
%   Y = LOEFFLER_DCT8_FIXED(X, CF) transforms every row of the integer N-by-8
%   matrix X exactly like rtl/loeffler_dct8.v: constants round(c*2^CF), each
%   rotation output formed from full-precision products and rounded once,
%       yp = floor((K*(xp+xq) + A*xq + 2^(CF-1)) / 2^CF).
%   The output has scale sqrt(8) and the LSB weight of the input.
%   Doubles hold every intermediate exactly (|values| < 2^40).

if nargin < 2, cf = 13; end
K   = ldct_constants(cf);
rnd = 2^(cf - 1);
sc  = 2^cf;
x   = double(X);

% stage 1
a0 = x(:,1) + x(:,8);   a7 = x(:,1) - x(:,8);
a1 = x(:,2) + x(:,7);   a6 = x(:,2) - x(:,7);
a2 = x(:,3) + x(:,6);   a5 = x(:,3) - x(:,6);
a3 = x(:,4) + x(:,5);   a4 = x(:,4) - x(:,5);

% stage 2
b0  = a0 + a3;   b3 = a0 - a3;
b1  = a1 + a2;   b2 = a1 - a2;
s47 = a4 + a7;   s56 = a5 + a6;
b4  = floor((K.O3_K * s47 + K.O3_A * a7 + rnd) / sc);
b7  = floor((K.O3_K * s47 - K.O3_B * a4 + rnd) / sc);
b5  = floor((K.O1_K * s56 + K.O1_A * a6 + rnd) / sc);
b6  = floor((K.O1_K * s56 - K.O1_B * a5 + rnd) / sc);

% stage 3
y0  = b0 + b1;   y4 = b0 - b1;
s23 = b2 + b3;
y2  = floor((K.E_K * s23 + K.E_A * b3 + rnd) / sc);
y6  = floor((K.E_K * s23 - K.E_B * b2 + rnd) / sc);
d4  = b4 + b6;   d6 = b4 - b6;
d7  = b7 + b5;   d5 = b7 - b5;

% stage 4
y1 = d7 + d4;    y7 = d7 - d4;
y3 = floor((K.R2 * d5 + rnd) / sc);
y5 = floor((K.R2 * d6 + rnd) / sc);

Y = [y0 y1 y2 y3 y4 y5 y6 y7];
end
