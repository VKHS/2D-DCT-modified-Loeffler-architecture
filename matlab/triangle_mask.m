function M = triangle_mask(zone)
%TRIANGLE_MASK Low-frequency triangle kept by the column-wise approximation.
%   M(v+1, u+1) is true for u + v <= zone (v vertical, u horizontal frequency).
[u, v] = meshgrid(0:7, 0:7);
M = (u + v) <= zone;
end
