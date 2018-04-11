CC=g++
CFLAGS=-Wall -I. -std=c++14 

all: test_qr_unblk qr_unblk_test

test_qr_unblk: test_qr_unblk.cpp hpc/matvec/qr.hpp
	$(CC) $(CFLAGS) -o test_qr_unblk test_qr_unblk.cpp

qr_unblk_test: qr_blk_test.cpp hpc/matvec/qr.hpp
	$(CC) $(CFLAGS) -o qr_unblk_test qr_blk_test.cpp

#lu_blk: lu_blk.cpp hpc/matvec/lu.hpp
#	$(CC) $(CFLAGS) -o lu_blk lu_blk.cpp

clean:
	rm test_qr_unblk
