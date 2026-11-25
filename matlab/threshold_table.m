function t = threshold_table(rf, unit)
%THRESHOLD_TABLE Integer SAD thresholds TINT(m+1), m = 0..15.
%   A row is replaced when SAD(Y_r, Y_ref) < TINT(m+1), which equals
%   mean(|X_r - X_ref|)/255 < m*unit for the orthonormal row DCT X.
if nargin < 2, unit = 0.001; end
scale = unit * 255 * 8 * sqrt(8) * 2^rf;
t = floor((0:15) * scale + 0.5);
end
