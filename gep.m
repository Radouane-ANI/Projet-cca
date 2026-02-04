function [L, U, P] = gep(A, droptol, thresh, udiag)
%GEP    Gaussian elimination with pivoting:
%       [L, U, P] = GEP(A, piv) computes the factorization P*A = L*U
%       of the m-by-n matrix A, where m >= n,
%       where L is m-by-n unit lower triangular, U is n-by-n upper triangular,
%
%       By itself, GEP(A) returns the final reduced matrix from the
%       elimination containing both L and U.
%       Reference:
%       N. J. Higham, Accuracy and Stability of Numerical Algorithms,
%       Second edition, Society for Industrial and Applied Mathematics,
%       Philadelphia, PA, 2002; chap. 9.
[m, n] = size(A);
if m < n, error('Matrix must be m-by-n with m >= n.'), end
if nargin < 2, droptol = 0.1; end
if nargin < 3, thresh = 1; end
if nargin < 4, udiag = 0; end
pp = 1:m; 

for k = 1:min(m-1,n)
    if thresh ~= 0
          % Find largest element in k'th column.
          [colmaxima, rowindices] = max( abs(A(k:m, k)) );
          row = rowindices(1)+k-1;
       % Permute largest element into pivot position.
       if A(k,k) < thresh * colmaxima 
           A( [k, row], : ) = A( [row, k], : );
           pp( [k, row] ) = pp( [row, k] );
       end
    end
    if A(k,k) == 0
      if udiag == 1
         A(k,k) = droptol;
      else % Zero pivot is problem only for no pivoting.
         error('Elimination breaks down with zero pivot.  Quitting...')
      end
    end
    i = k+1:m;
    A(i,k) = A(i,k)/A(k,k);          % Multipliers.
    if k+1 <= n
       % Elimination
       j = k+1:n;
       A(i,j) = A(i,j) - A(i,k) * A(k,j);
       A(i,j) = A(i,j) .* (abs(A(i,j)) >= droptol);
    end
end
if nargout <= 1
   L = A;
   return
end
L = tril(A,-1) + eye(m,n);
U = triu(A);
U = U(1:n,:);
if nargout >= 3, P = eye(m); P = P(pp,:); end
