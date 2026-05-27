function [results] = test_ilu(A, L, U, P)
if nargin < 4
    P = sparse(eye(size(A)));
end
n = size(A, 1);
b = A * ones(n, 1);

residual_matrix = (P * A) - (L * U);
results.error_fro = norm(residual_matrix, 'fro');
results.relative_error = results.error_fro / norm(A, 'fro');
results.fill_in = (nnz(L) + nnz(U)) / nnz(A);

tol = 1e-10; % Tolérance d'arrêt du solveur
maxit = 200;

try
    [x, flag, ~, ~, resvec] = gmres(P*A, P*b, [], tol, maxit, L, U);
    results.resvec = resvec;
    results.iterations = length(resvec) - 1;
    results.flag = flag; % 0 = succès, 1 = échec, etc.
    
    % Check for spurious convergence (numerical artifact)
    true_error = norm(x - ones(n, 1)) / norm(ones(n, 1));
    if true_error > 1e-4
        results.flag = 1; % Force flag to 1 (failure)
    end
catch e
    % Si la factorisation a produit des NaN/Inf, gmres peut crasher
    results.resvec = [];
    results.iterations = maxit; % On pénalise avec maxit
    results.flag = 1;
end
end