set terminal png size 900, 500 background rgb 'white'
set output "unblk.png"
set xlabel "Matrix Dimension A: M=N"
set ylabel "MFLOPS"
set title "Ungeblockte QR-Zerlegung"
set key outside
set pointsize 0.5
plot "unblk.data" using 1:5 with linespoints lt 2 lw 1 title "unblk", \
     "unblk.data" using 1:8 with linespoints lt 3 lw 1 title "unblk ref"
