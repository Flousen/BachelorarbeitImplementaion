clear all; close all; clc;

A = magic(20);
[m,n] = size(A)
k=min(m,n)

ib=5;

for i = 1:ib:k-ib
  qr(A(i:m,i:i+ib-1));
  
  DLARFT
  
  
  
end

qr( A( i:end , i:n ) );