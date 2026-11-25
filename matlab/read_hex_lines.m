function lines = read_hex_lines(path)
%READ_HEX_LINES Non-empty lines of a hex vector file as a cell array.
fid = fopen(path, 'r');
if fid < 0
    error('read_hex_lines:open', 'cannot open %s', path);
end
lines = {};
while true
    l = fgetl(fid);
    if ~ischar(l), break; end
    l = strtrim(l);
    if ~isempty(l)
        lines{end + 1} = l; %#ok<AGROW>
    end
end
fclose(fid);
end
