function Q = jpeg_qtable(quality)
%JPEG_QTABLE ITU-T T.81 Annex K luminance table with IJG quality scaling.
T = [16 11 10 16  24  40  51  61; 12 12 14 19  26  58  60  55;
     14 13 16 24  40  57  69  56; 14 17 22 29  51  87  80  62;
     18 22 37 56  68 109 103  77; 24 35 55 64  81 104 113  92;
     49 64 78 87 103 121 120 101; 72 92 95 98 112 100 103  99];
q = min(max(round(quality), 1), 100);
if q < 50, s = floor(5000 / q); else, s = 200 - 2 * q; end
Q = min(max(floor((T * s + 50) / 100), 1), 255);
end
