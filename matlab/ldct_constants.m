function K = ldct_constants(cf)
%LDCT_CONSTANTS Rotation constants of the 8-point modified Loeffler engine.
%   K = LDCT_CONSTANTS() returns the floating-point constants.
%   K = LDCT_CONSTANTS(CF) returns round(c * 2^CF), bit-exact with the RTL
%   (rtl/loeffler_dct8.v) and HLS code (hls/dct2d_loeffler.cpp).
%
%   Every rotation of the engine uses the 3-multiplier form
%       z = K*(xp + xq),   yp = z + A*xq,   yq = z - B*xp
%
%   name   value                              rotation
%   E_*    sqrt2*cos(6pi/16), ...             even part, stage 3  (Y2, Y6)
%   O3_*   cos(3pi/16), ...                   odd part,  stage 2  (b4, b7)
%   O1_*   cos(pi/16), ...                    odd part,  stage 2  (b5, b6)
%   R2     sqrt2                              odd part,  stage 4  (Y3, Y5)

c  = cos((0:7) * pi / 16);          % c(k+1) = cos(k*pi/16)
r2 = sqrt(2);
K.E_K  = r2 * c(7);                 % sqrt2*cos(6pi/16)
K.E_A  = r2 * (c(3) - c(7));        % sqrt2*(cos(2pi/16) - cos(6pi/16))
K.E_B  = r2 * (c(3) + c(7));        % sqrt2*(cos(2pi/16) + cos(6pi/16))
K.O3_K = c(4);                      % cos(3pi/16)
K.O3_A = c(6) - c(4);               % cos(5pi/16) - cos(3pi/16)
K.O3_B = c(4) + c(6);               % cos(3pi/16) + cos(5pi/16)
K.O1_K = c(2);                      % cos(pi/16)
K.O1_A = c(8) - c(2);               % cos(7pi/16) - cos(pi/16)
K.O1_B = c(2) + c(8);               % cos(pi/16) + cos(7pi/16)
K.R2   = r2;                        % sqrt2

if nargin >= 1 && ~isempty(cf)
    f = fieldnames(K);
    for i = 1:numel(f)
        K.(f{i}) = round(K.(f{i}) * 2^cf);
    end
end
end
