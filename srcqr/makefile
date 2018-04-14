CC=g++
CFLAGS=-Wall -I. -std=c++14 

all: test_qr queue_test

test_qr: test_qr.cpp hpc/matvec/qr.hpp
	$(CC) $(CFLAGS) -o $@ test_qr.cpp

queue_test: queue_test.cpp hpc/matvec/qr.hpp
	$(CC) $(CFLAGS) -o $@ queue_test.cpp

clean:
	rm test_qr_unblk
