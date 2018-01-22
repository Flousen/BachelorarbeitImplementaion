function [A,R] = Householder(A)

for j = 1:size(A,2)
    [v,beta(j)] = HouseholderVector( A(j:end,j) );
    A(j:end,j:end) = A(j:end,j:end) -( beta(j) * v *  v' * A(j:end,j:end) );
    if j < size(A,1)
       A(j+1:end,j) = v(2:size(A,1)-j+1); 
    end
end
if nargout == 2
    R = A;
    A = eye(size(A,1));
    for j = size(R,2):-1:1
        v = [1; R(j+1:end,j)];
        A(j:end,j:end) = A(j:end,j:end) - ( beta(j) * v * v' * A(j:end,j:end) );
    end
    R = triu(R);
end