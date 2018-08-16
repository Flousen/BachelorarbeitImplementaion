set terminal svg size 900, 500 background rgb 'white'
set output "img/blk_vlg.svg"
set xlabel "Matrix dim A: M=N"
set title "blk"
set key outside
set pointsize 0.5
plot "dat/blk.data" using 1:5 with linespoints lt 1 lw 3 title "blk", \
     "dat/blklarftref.data" using 1:5 with linespoints lt 2 lw 3 title "blk larft ref", \
     "dat/blklarfbref.data" using 1:5 with linespoints lt 3 lw 3 title "blk larfb ref", \
     "dat/blkboth.data" using 1:5 with linespoints lt 5 lw 3 title "blk both ref", \
     "dat/blk.data" using 1:8 with linespoints lt 4 lw 3 title "blk ref"
