function [blocks, F, status] = read_vectors(dirname)
%READ_VECTORS Load a golden vector set (vectors/<mode>/).
%   BLOCKS(r+1, k+1, n)  pixels, from input_rows.hex (pixel 0 in the last 2 digits)
%   F(v+1, u+1, n)       expected coefficients, from expected_cols.hex
%   STATUS(n)            {approx, replaced[7:0]}, from expected_status.hex

rows = read_hex_lines(fullfile(dirname, 'input_rows.hex'));
cols = read_hex_lines(fullfile(dirname, 'expected_cols.hex'));
st   = read_hex_lines(fullfile(dirname, 'expected_status.hex'));
n = numel(st);
blocks = zeros(8, 8, n, 'uint8');
F      = zeros(8, 8, n);
status = zeros(n, 1);
for b = 1:n
    for r = 1:8
        line = rows{8 * (b - 1) + r};
        for k = 1:8
            pos = 16 - 2 * k + 1;
            blocks(r, k, b) = hex2dec(line(pos:pos + 1));
        end
    end
    for u = 1:8
        line = cols{8 * (b - 1) + u};
        for v = 1:8
            pos = 24 - 3 * v + 1;
            val = hex2dec(line(pos:pos + 2));
            if val >= 2048, val = val - 4096; end
            F(v, u, b) = val;
        end
    end
    status(b) = hex2dec(st{b});
end
end
