load("data/494_bus.mat");
M = Problem.A;
addpath('~/chop');
seuils = logspace(-5, -1, 8);
n_tests = length(seuils);

% Initialisation des résultats
results_mixte = struct('iters', zeros(n_tests, 1), 'fill_in', zeros(n_tests, 1), ...
    'rel_err', zeros(n_tests, 1), 'precision_dist', zeros(n_tests, 5));
results_std = struct('iters', zeros(n_tests, 1), 'fill_in', zeros(n_tests, 1), 'rel_err', zeros(n_tests, 1));
results_ilu = struct('iters', zeros(n_tests, 1), 'fill_in', zeros(n_tests, 1), 'rel_err', zeros(n_tests, 1));

fprintf('===== COMPARAISON 3 VERSIONS =====\n\n');
fprintf('Seuil    | Mix Iter | Std Iter | ILU Iter | %%16-bit | %%24-bit | %%32-bit | %%48-bit | %%64-bit\n');
fprintf('---------|----------|----------|----------|---------|---------|---------|---------|---------\n');

for i = 1:n_tests
    s = seuils(i);

    % ===== 1. TEST GEP MIXTE =====
    % gep_mixte renvoie maintenant les compteurs réels de précision !
    [L_mix, U_mix, P_mix, counts] = gep_mixte(M, s, 1, 1);
    res_mix = test_ilu(M, L_mix, U_mix, P_mix);
    results_mixte.fill_in(i) = (nnz(L_mix) + nnz(U_mix)) / nnz(M);
    results_mixte.iters(i) = res_mix.iterations;
    results_mixte.rel_err(i) = res_mix.relative_error;

    % On utilise DIRECTEMENT les compteurs de la fonction, pas de déduction hasardeuse !
    results_mixte.precision_dist(i, :) = counts;
    tot_processed = sum(counts);
    pct = (counts / tot_processed) * 100;

    % ===== 2. TEST GEP STANDARD (64-bits) =====
    [L_std, U_std, P_std] = gep(M, s, 1, 1);
    res_std = test_ilu(M, L_std, U_std, P_std);
    results_std.fill_in(i) = (nnz(L_std) + nnz(U_std)) / nnz(M);
    results_std.iters(i) = res_std.iterations;
    results_std.rel_err(i) = res_std.relative_error;

    % ===== 3. TEST ILU MATLAB =====
    setup.type = 'ilutp';
    setup.droptol = s;
    setup.udiag = 1;
    [L_ilu, U_ilu, P_ilu] = ilu(M, setup);
    res_ilu = test_ilu(M, L_ilu, U_ilu, P_ilu);
    results_ilu.fill_in(i) = (nnz(L_ilu) + nnz(U_ilu)) / nnz(M);
    results_ilu.iters(i) = res_ilu.iterations;
    results_ilu.rel_err(i) = res_ilu.relative_error;

    fprintf('%.1e | %8d | %8d | %8d | %7.1f%% | %7.1f%% | %7.1f%% | %7.1f%% | %7.1f%%\n', ...
        s, res_mix.iterations, res_std.iterations, res_ilu.iterations, ...
        pct(1), pct(2), pct(3), pct(4), pct(5));
end

fprintf('\n===== GRAPHIQUES GÉNÉRÉS =====\n');
c_mix = '#0072BD';
c_std = '#77AC30';
c_ilu = '#D95319';

% --- GRAPHIQUE 1 & 2 : Convergence et Fill-in
figure('Name', 'Convergence et Fill-in', 'Position', [100 100 1200 400]);

subplot(1,2,1);
loglog(seuils, results_mixte.iters, '-o', 'LineWidth', 2, 'Color', c_mix); hold on;
loglog(seuils, results_std.iters, '-^', 'LineWidth', 2, 'Color', c_std);
loglog(seuils, results_ilu.iters, '-s', 'LineWidth', 2, 'Color', c_ilu);
set(gca, 'XDir', 'reverse');
grid on;
xlabel('Drop-tolerance');
ylabel('Itérations GMRES');
legend('GEP Mixte', 'GEP Std (64b)', 'ILU MATLAB', 'Location', 'best');
title('Convergence');

subplot(1,2,2);
semilogx(seuils, results_mixte.fill_in, '-o', 'LineWidth', 2, 'Color', c_mix); hold on;
semilogx(seuils, results_std.fill_in, '-^', 'LineWidth', 2, 'Color', c_std);
set(gca, 'XDir', 'reverse');
grid on;
xlabel('Drop-tolerance');
ylabel('Facteur de Fill-in');
title('Évolution du Fill-in');

% --- GRAPHIQUE 3 : Répartition adaptative des précisions
figure('Name', 'Distribution des formats de précision');
dist_pct = results_mixte.precision_dist ./ sum(results_mixte.precision_dist, 2) * 100;
b = bar(1:n_tests, dist_pct, 'stacked', 'FaceColor', 'flat');
b(1).CData = [0.93 0.69 0.13]; % Jaune/Or (16-bits)
b(2).CData = [0.85 0.33 0.10]; % Orange (24-bits)
b(3).CData = [0.30 0.75 0.93]; % Cyan (32-bits)
b(4).CData = [0.47 0.67 0.19]; % Vert (48-bits)
b(5).CData = [0 0.45 0.74];    % Bleu foncé (64-bits)
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.1e', x), seuils, 'UniformOutput', false));
xtickangle(45);
ylabel('Proportion des éléments (%)');
xlabel('Drop-tolerance');
legend('FP16 (Half)', 'FP24', 'FP32 (Single)', 'FP48', 'FP64 (Double)', 'Location', 'best');
title('Allocation dynamique de la précision matérielle (GEP Mixte)');
grid on;

% --- GRAPHIQUE 4 : Empreinte Mémoire Théorique
% Calcul en Kilo-Octets (KB) des éléments non nuls (valeurs seules)
mem_std = (results_std.fill_in * nnz(M)) * 8 / 1024; % 8 bytes par élément en Double
mem_mix = (results_mixte.precision_dist(:,1)*2 + ...
    results_mixte.precision_dist(:,2)*3 + ...
    results_mixte.precision_dist(:,3)*4 + ...
    results_mixte.precision_dist(:,4)*6 + ...
    results_mixte.precision_dist(:,5)*8) / 1024;

figure('Name', 'Empreinte Mémoire');
semilogx(seuils, mem_mix, '-o', 'LineWidth', 2.5, 'Color', c_mix); hold on;
semilogx(seuils, mem_std, '-^', 'LineWidth', 2.5, 'Color', c_std);
set(gca, 'XDir', 'reverse');
xlabel('Drop-tolerance');
ylabel('Taille en Kilo-Octets (KB)');
legend('Stockage Mixte (Compressé)', 'Stockage Standard (FP64)', 'Location', 'best');
title('Empreinte Mémoire Théorique du Préconditionneur');
grid on;

%%
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
val_out = val_in;
u_16 = 4.88e-4;
u_24 = 1.53e-5;
u_32 = 5.96e-8;
u_48 = 7.28e-12;
fp16_max = 65500;

mask_16 = (tau >= u_16) & (abs(val_in) <= fp16_max);
mask_24 = (tau >= u_24) & ~mask_16;
mask_32 = (tau >= u_32) & ~mask_16 & ~mask_24;
mask_48 = (tau >= u_48) & ~mask_16 & ~mask_24 & ~mask_32;
mask_64 = ~mask_16 & ~mask_24 & ~mask_32 & ~mask_48;

n_16 = sum(mask_16);
n_24 = sum(mask_24);
n_32 = sum(mask_32);
n_48 = sum(mask_48);
n_64 = sum(mask_64);

if n_16 > 0
    opt.format = 'h';
    val_out(mask_16) = chop(val_in(mask_16), opt);
end

if n_24 > 0
    opt.format = 'custom';
    opt.params = [16, 127];
    val_out(mask_24) = chop(val_in(mask_24), opt);
end

if n_32 > 0
    val_out(mask_32) = double(single(val_in(mask_32)));
end

if n_48 > 0
    opt.format = 'custom';
    opt.params = [38, 1023];
    val_out(mask_48) = chop(val_in(mask_48), opt);
end
end
