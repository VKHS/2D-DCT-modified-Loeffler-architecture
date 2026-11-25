function [Y, stages] = loeffler_dct8(X)
%LOEFFLER_DCT8 8-point Loeffler DCT of every row of X (output scale sqrt(8)).
%   Y = LOEFFLER_DCT8(X), X of size N-by-8, computes Y = sqrt(8) * X * C8.'
%   with the 4-stage Loeffler data flow (11 multiplications, 29 additions).
%   Y / sqrt(8) is the orthonormal DCT-II of each row.
%
%   [Y, STAGES] = LOEFFLER_DCT8(X) also returns the outputs of every stage.

K = ldct_constants();
x = double(X);

% stage 1: input butterflies                                      8 add
a0 = x(:,1) + x(:,8);   a7 = x(:,1) - x(:,8);
a1 = x(:,2) + x(:,7);   a6 = x(:,2) - x(:,7);
a2 = x(:,3) + x(:,6);   a5 = x(:,3) - x(:,6);
a3 = x(:,4) + x(:,5);   a4 = x(:,4) - x(:,5);

% stage 2: even butterflies, odd rotations 3pi/16 and pi/16   4+6 add, 6 mul
b0 = a0 + a3;   b3 = a0 - a3;
b1 = a1 + a2;   b2 = a1 - a2;
z  = K.O3_K * (a4 + a7);   b4 = z + K.O3_A * a7;   b7 = z - K.O3_B * a4;
z  = K.O1_K * (a5 + a6);   b5 = z + K.O1_A * a6;   b6 = z - K.O1_B * a5;

% stage 3: even Y0/Y4 and sqrt2-rotation 6pi/16, odd butterflies  9 add, 3 mul
y0 = b0 + b1;   y4 = b0 - b1;
z  = K.E_K * (b2 + b3);    y2 = z + K.E_A * b3;    y6 = z - K.E_B * b2;
d4 = b4 + b6;   d6 = b4 - b6;
d7 = b7 + b5;   d5 = b7 - b5;

% stage 4: odd outputs                                        2 add, 2 mul
y1 = d7 + d4;   y7 = d7 - d4;
y3 = K.R2 * d5; y5 = K.R2 * d6;

Y = [y0 y1 y2 y3 y4 y5 y6 y7];
if nargout > 1
    stages = struct('stage1', [a0 a1 a2 a3 a4 a5 a6 a7], ...
                    'stage2', [b0 b1 b2 b3 b4 b5 b6 b7], ...
                    'stage3', [y0 y4 y2 y6 d4 d5 d6 d7], ...
                    'stage4', Y);
end
end
