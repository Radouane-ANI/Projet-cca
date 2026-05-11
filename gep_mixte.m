function [L, U, P, counts] = gep_mixte(A, droptol, thresh, udiag)
[m, n] = size(A);
if m < n, error('Matrix must be m-by-n with m >= n.'), end
if nargin < 2, droptol = 0.1; end
if nargin < 3, thresh = 1; end
if nargin < 4, udiag = 0; end

pp = 1:m;
col_norms = vecnorm(A);
chop([], struct('format', 'h'));
counts = zeros(1, 5); % [nb_fp16, nb_fp24, nb_fp32, nb_fp48, nb_fp64]

for k = 1:min(m-1, n)
    if thresh ~= 0
        [colmaxima, rowindices] = max(abs(A(k:m, k)));
        row = rowindices(1) + k - 1;
        if abs(A(k,k)) < thresh * colmaxima
            A([k, row], :) = A([row, k], :);
            pp([k, row]) = pp([row, k]);
        end
    end

    if A(k,k) == 0
        if udiag == 1
            A(k,k) = droptol;
        else
            error('Breakdown zero pivot.');
        end
    end

    i = k+1:m;
    multipliers = A(i,k) / A(k,k);

    drop_mask_L = abs(multipliers) < (droptol * col_norms(k) / abs(A(k,k)));
    multipliers(drop_mask_L) = 0;

    kept_mask_L = ~drop_mask_L;
    if any(kept_mask_L)
        tau_L = (droptol * col_norms(k)) ./ abs(multipliers(kept_mask_L) * A(k,k));
        [multipliers(kept_mask_L), c_16, c_24, c_32, c_48, c_64] = apply_mixed_precision(multipliers(kept_mask_L), tau_L);
        counts = counts + [c_16, c_24, c_32, c_48, c_64];
    end

    A(i,k) = multipliers;

    if k+1 <= n
        j = k+1:n;
        A(i,j) = A(i,j) - A(i,k) * A(k,j);

        row_U = A(k, j);
        drop_mask_U = abs(row_U) < (droptol * col_norms(j));
        row_U(drop_mask_U) = 0;

        kept_mask_U = ~drop_mask_U & (row_U ~= 0);
        if any(kept_mask_U)
            norms_U = col_norms(j(kept_mask_U));
            tau_U = (droptol * norms_U) ./ abs(row_U(kept_mask_U));
            [row_U(kept_mask_U), c_16, c_24, c_32, c_48, c_64] = apply_mixed_precision(row_U(kept_mask_U), tau_U);
            counts = counts + [c_16, c_24, c_32, c_48, c_64];
        end

        A(k, j) = row_U;

        tau_pivot = (droptol * col_norms(k)) / abs(A(k,k));
        [A(k,k), c_16p, c_24p, c_32p, c_48p, c_64p] = apply_mixed_precision(A(k,k), tau_pivot);
        counts = counts + [c_16p, c_24p, c_32p, c_48p, c_64p];
    end
end

if nargout <= 1
    L = A; return;
end

L = tril(A,-1) + eye(m,n);
U = triu(A);
U = U(1:n,:);
if nargout >= 3, P = eye(m); P = P(pp,:); end
end

function [val_out, n_16, n_24, n_32, n_48, n_64] = apply_mixed_precision(val_in, tau)
    val_in = full(val_in); % Ensure it's dense for operations
    val_out = val_in;
    
    % Unités d'arrondi (u = 2^(-t))
    u_16 = 4.88e-4;   % Half (t=11)
    u_24 = 1.53e-5;   % Custom 24-bit (t=17)
    u_32 = 5.96e-8;   % Single (t=24)
    u_48 = 1.45e-11;  % Custom 48-bit (t=37) -> 2^-36 ≈ 1.45e-11
    
    % Limites matérielles (emax)
    fp16_max = 65504; 
    fp24_max = 3.4e38; % Si emax reste 127 (souvent le cas en simul logicielle)
    fp32_max = double(realmax('single'));
    
    % Création des masques (Logique de cascade)
    abs_val = abs(val_in);
    mask_16 = (tau >= u_16) & (abs_val <= fp16_max);
    mask_24 = (tau >= u_24) & (abs_val <= fp24_max) & ~mask_16;
    mask_32 = (tau >= u_32) & (abs_val <= fp32_max) & ~mask_16 & ~mask_24;
    mask_48 = (tau >= u_48) & ~mask_16 & ~mask_24 & ~mask_32; % Range identique au fp64
    mask_64 = ~mask_16 & ~mask_24 & ~mask_32 & ~mask_48;

    % Application de chop avec structures propres
    if any(mask_16)
        val_out(mask_16) = chop(val_in(mask_16), struct('format', 'h'));
    end

    if any(mask_24)
        % t=17 car 16 bits stockés + 1 caché. emax=127 pour range type float32
        val_out(mask_24) = chop(val_in(mask_24), struct('format', 'c', 'params', [17, 127]));
    end

    if any(mask_32)
        val_out(mask_32) = single(val_in(mask_32)); 
    end

    if any(mask_48)
        % t=37 car 36 bits stockés + 1 caché. emax=1023 (double range)
        val_out(mask_48) = chop(val_in(mask_48), struct('format', 'c', 'params', [37, 1023], 'round', 4));
    end

    % Comptage final
    n_16 = sum(mask_16); n_24 = sum(mask_24); n_32 = sum(mask_32); 
    n_48 = sum(mask_48); n_64 = sum(mask_64);
end
