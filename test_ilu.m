function [results] = test_ilu(A, L, U)
n = size(A, 1);
b = A * ones(n, 1);

residual_matrix = A - (L * U);
results.error_fro = norm(residual_matrix, 'fro');
results.relative_error = results.error_fro / norm(A, 'fro');
results.fill_in = (nnz(L) + nnz(U)) / nnz(A);

tol = 1e-10; % Tolérance d'arrêt du solveur
maxit = 200;
%L = P'*L;
[~, flag, ~, ~, resvec] = gmres(A, b, [], tol, maxit, L, U);

results.resvec = resvec;
results.iterations = length(resvec) - 1;
results.flag = flag; % 0 = succès, 1 = échec
end