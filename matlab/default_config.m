function cfg = default_config()
%DEFAULT_CONFIG Word lengths and TSC settings (identical to RTL/HLS/Python).
%   Exact fixed-point transform: cfg.tsc_enable = false; cfg.approx_mode = 0;

cfg.cf          = 13;     % fractional bits of the rotation constants
cfg.rf          = 2;      % guard bits carried from the row to the column pass
cfg.tsc_enable  = true;   % stage 1: replace near-identical consecutive rows
cfg.t0          = 1;      % default threshold index       (x 0.001)
cfg.step        = 1;      % adaptive increment            (x 0.001)
cfg.tmax        = 10;     % maximum threshold index       (x 0.001)
cfg.min_similar = 3;      % similar pairs among rows 0..3 that trigger stage 2
cfg.approx_mode = 1;      % 0 off, 1 TSC-triggered, 2 always
cfg.zone        = 5;      % triangular zone: keep u + v <= zone
cfg.unit        = 0.001;  % threshold unit
end
