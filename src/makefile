CC=g++
CFLAGS=-Wall -I. -std=c++14 

all: test_qr_unblk qr_unblk_test

test_qr_unblk: test_qr_unblk.cpp hpc/matvec/qr.hpp
	$(CC) $(CFLAGS) -o test_qr_unblk test_qr_unblk.cpp

qr_unblk_test: qr_unblk_test.cpp hpc/matvec/qr.hpp
	$(CC) $(CFLAGS) -o qr_unblk_test qr_unblk_test.cpp

clean:
	rm test_qr_unblk
