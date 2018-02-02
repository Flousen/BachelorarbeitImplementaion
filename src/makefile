CC=g++
CFLAGS=-Wall -I. -std=c++14 -O3

test_qr_unblk: test_qr_unblk.cpp hpc/matvec/qr.hpp
	$(CC) $(CFLAGS) -o test_qr_unblk test_qr_unblk.cpp


clean:
	rm test_qr_unblk
