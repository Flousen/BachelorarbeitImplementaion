set terminal png size 900, 500 background rgb 'white'
set output "img/blkbs.png"
set xlabel "Matrix Dimension A: M=N"
set ylabel "MFLOPS"
set title "Geblockte QR-Zerlegung"
set key below 
set pointsize 0.5
plot "dat/blkbs.data" using 1:3 with linespoints lt 1 lw 1 pt 1 ps 1.5 title  "bs =  8", \
     "dat/blkbs.data" using 1:4 with linespoints lt 2 lw 1 pt 2 ps 1.5 title  "bs = 16", \
     "dat/blkbs.data" using 1:5 with linespoints lt 3 lw 1 pt 3 ps 1.5 title  "bs = 32", \
     "dat/blkbs.data" using 1:6 with linespoints lt 4 lw 1 pt 4 ps 1.5 title  "bs = 64", \
     "dat/blkbs.data" using 1:7 with linespoints lt 5 lw 1 pt 5 ps 1.5 title  "bs =128", \
     "dat/blkbs.data" using 1:8 with line lw 3 lc rgb "red"title "Theo. Peak Performance"
#     "dat/blkbs.data" using 1:8 with linespoints lt 4 lw 1 pt 4 ps 1.5 title  "bs = 48", \
#     "dat/blkbs.data" using 1:9 with linespoints lt 4 lw 1 pt 4 ps 1.5 title  "bs = 56", \
#     "dat/blkbs.data" using 1:10 with linespoints lt 4 lw 1 pt 4 ps 1.5 title  "bs = 64", \
