set terminal png size 900, 500 background rgb 'white'
set output "unblk.png"
set xlabel "Matrix Dimension A: M=N"
set ylabel "MFLOPS"
set title "Ungeblockte QR-Zerlegung"
set key below
set pointsize 0.5
plot "unblk.data" using 1:5 with linespoints lt 2 lw 1 pt 8 ps 1.5 title "Eigene dgeqrf2", \
     "unblk.data" using 1:8 with linespoints lt 3 lw 1 pt 2 ps 1.5 title "MKL dgeqrf2"#, \
#     "blk.data" using 1:9 with line lw 3 lc rgb "red"title "Theoretical Peak Performance"
