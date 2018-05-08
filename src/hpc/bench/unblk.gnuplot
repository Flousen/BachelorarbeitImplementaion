set terminal svg size 900, 500 background rgb 'white'
set output "unblk.svg"
set xlabel "Matrix dim A: M=N"
set title "unblk"
set key outside
set pointsize 0.5
plot "unblk.data" using 1:5 with linespoints lt 2 lw 3 title "unblk", \
     "unblk.data" using 1:8 with linespoints lt 3 lw 3 title "unblk ref"
