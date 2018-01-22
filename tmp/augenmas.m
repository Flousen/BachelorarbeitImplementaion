clear all; close all; clc;

A = magic(5);

[Q,R] = qr(A);

norm(Q*R - A) ;

[Qe,Re] = Householder(A);

norm(Qe*Re - A) ;

Q
Qe
R
Re


