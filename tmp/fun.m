n=5
A=rand(n,n)

V=qr(A)

for i=1:5
  V(i,i)=1
end
for i=1:5
  V(i:n,1:i-1)'
  V(i:n,i)
 T(1:i-1,i) = V(i:n,1:i-1)' * V(i:n,i)
 T(1:i-1,i) = V(1:i-1,i:n) * V(i,i:n)'
end