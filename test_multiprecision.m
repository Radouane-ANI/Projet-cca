load("data/494_bus.mat");
M = Problem.A;
addpath('~/chop');
seuils = logspace(-10, -1, 8);
n_tests = length(seuils);
%{
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
%}

% ===============================================
% TEST COMPARATIF : GEP MIXTE vs ILU FULL 16/32/64
% ===============================================

% Structures pour stocker les résultats [Mixte, ILU64, ILU32, ILU16]
iters = zeros(n_tests, 4); 
mem = zeros(n_tests, 4);   

fprintf('===== COMPARAISON DES APPROCHES (Troncature sur ILU MATLAB) =====\n\n');
fprintf('Seuil    | Mix Iter | ILU64 Iter | ILU32 Iter | ILU16 Iter \n');
fprintf('---------|----------|------------|------------|------------\n');

% Configuration pour le FP16
opt_half = struct('format', 'h');

for i = 1:n_tests
    s = seuils(i);
    
    % --- 1. GEP Mixte (Adaptatif pendant la factorisation) ---
    [L_mix, U_mix, P_mix, counts] = gep_mixte(M, s, 1, 1);
    res_mix = test_ilu(M, L_mix, U_mix, P_mix);
    iters(i, 1) = res_mix.iterations;
    % Mémoire Mixte : pondérée par les 5 formats
    mem(i, 1) = (counts(1)*2 + counts(2)*3 + counts(3)*4 + counts(4)*6 + counts(5)*8) / 1024;
    
    % --- FACTORISATION ILU DE BASE ---
    setup.type = 'ilutp'; 
    setup.droptol = s; 
    setup.udiag = 1;
    [L_ilu, U_ilu, P_ilu] = ilu(M, setup);
    
    % Nombre d'éléments réels stockés par ILU
    nnz_total = nnz(L_ilu) + nnz(U_ilu) - size(M,1); 
    
    % --- 2. ILU FP64 (Baseline Standard) ---
    res_64 = test_ilu(M, L_ilu, U_ilu, P_ilu);
    iters(i, 2) = res_64.iterations;
    mem(i, 2) = (nnz_total * 8) / 1024;
    
    % --- 3. ILU FP32 (Troncature a posteriori) ---
    % spfun applique la conversion uniquement sur les éléments non nuls
    L_32 = spfun(@(x) double(single(x)), L_ilu);
    U_32 = spfun(@(x) double(single(x)), U_ilu);
    res_32 = test_ilu(M, L_32, U_32, P_ilu);
    iters(i, 3) = res_32.iterations;
    mem(i, 3) = (nnz_total * 4) / 1024;
    
    % --- 4. ILU FP16 (Troncature a posteriori) ---
    L_16 = spfun(@(x) chop(full(x), opt_half), L_ilu);
    U_16 = spfun(@(x) chop(full(x), opt_half), U_ilu);
    res_16 = test_ilu(M, L_16, U_16, P_ilu);
    iters(i, 4) = res_16.iterations;
    mem(i, 4) = (nnz_total * 2) / 1024;
    
    fprintf('%.1e | %8d | %10d | %10d | %10d \n', s, iters(i,1), iters(i,2), iters(i,3), iters(i,4));
end

% =========================================================================
% 1. TEST DE ROBUSTESSE (Qualité de la convergence)
% =========================================================================
figure('Name', 'Test de Robustesse', 'Position', [100, 100, 800, 500]);

% On remplace les 0 itérations (crash) par des NaN pour couper les courbes
iters_plot = iters;
iters_plot(iters_plot == 0) = NaN;

loglog(seuils, iters_plot(:,1), '-o', 'LineWidth', 2.5, 'Color', '#0072BD', 'MarkerSize', 8); hold on;
loglog(seuils, iters_plot(:,2), '-^', 'LineWidth', 2, 'Color', '#77AC30');
loglog(seuils, iters_plot(:,3), '-s', 'LineWidth', 2, 'Color', '#D95319');
loglog(seuils, iters_plot(:,4), '-d', 'LineWidth', 2, 'Color', '#EDB120');

set(gca, 'XDir', 'reverse');
grid on;
xlabel('Drop-tolerance (Sévérité du filtre)');
ylabel('Nombre d''itérations GMRES');
title('Test de Robustesse : Survie du solveur selon la précision');
legend('GEP Mixte Adaptatif', 'ILU FP64 (Référence)', 'ILU FP32', 'ILU FP16 (Instable)', 'Location', 'best');

% =========================================================================
% 2. PROFIL DE PERFORMANCE (Empreinte Mémoire)
% =========================================================================
% 1. Construction de la matrice des coûts (Mémoire)
% Si l'algorithme a crashé (0 itération), son coût devient Infini
cost_matrix = mem;
cost_matrix(iters == 0) = Inf;

% 2. Calcul du meilleur coût pour chaque test (droptol)
min_cost = min(cost_matrix, [], 2);

% 3. Calcul du ratio de performance (tau)
% ratio = 1 signifie que l'algorithme est le meilleur sur ce test
perf_ratio = cost_matrix ./ min_cost;

% 4. Préparation de l'axe X (le facteur de surcoût tau)
tau_max = max(perf_ratio(perf_ratio < Inf));
if isempty(tau_max) || tau_max == 1
    tau_max = 5; % Sécurité graphique
end
% On crée un axe log pour tau, de 1 (meilleur) jusqu'au pire facteur
tau_vals = linspace(1, tau_max * 1.1, 1000); 

% 5. Calcul des courbes du profil (Fraction des tests résolus <= tau)
n_solvers = 4;
profile = zeros(length(tau_vals), n_solvers);
for s = 1:n_solvers
    for k = 1:length(tau_vals)
        % Combien de tests ce solveur a-t-il résolu avec un coût <= tau * min_cost ?
        profile(k, s) = sum(perf_ratio(:, s) <= tau_vals(k)) / n_tests;
    end
end

% 6. Tracé du Profil de Performance
figure('Name', 'Profil de Performance ', 'Position', [150, 150, 800, 500]);

% Utilisation de lignes distinctes pour un rendu "Publication"
plot(tau_vals, profile(:,1), '-',  'LineWidth', 3,   'Color', '#0072BD'); hold on; % Mixte
plot(tau_vals, profile(:,2), '--', 'LineWidth', 2.5, 'Color', '#77AC30');          % FP64
plot(tau_vals, profile(:,3), '-.', 'LineWidth', 2.5, 'Color', '#D95319');          % FP32
plot(tau_vals, profile(:,4), ':',  'LineWidth', 2.5, 'Color', '#EDB120');          % FP16

grid on;
xlabel('Facteur de surcoût toléré par rapport au meilleur (\tau)');
ylabel('Fraction des tests résolus avec succès');
title('Profil de Performance (Dolan-Moré) : Efficacité Mémoire');
legend('GEP Mixte Adaptatif', 'ILU FP64', 'ILU FP32', 'ILU FP16', 'Location', 'southeast');
ylim([0 1.05]);
xlim([1 max(tau_vals)]);
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
    
    % Unités d'arrondi (u = 2^(-t))
    u_16 = 4.88e-4;   % Half (t=11)
    u_24 = 1.53e-5;   % Custom 24-bit (t=17)
    u_32 = 5.96e-8;   % Single (t=24)
    u_48 = 1.45e-11;  % Custom 48-bit (t=37) -> 2^-36 ≈ 1.45e-11
    
    % Limites matérielles (emax)
    fp16_max = 65504; 
    fp24_max = 3.4e38; % Si emax reste 127 (souvent le cas en simul logicielle)
    fp32_max = realmax('single');
    
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