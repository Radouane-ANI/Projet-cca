% =========================================================================
% BENCHMARK MULTIPRÉCISION : GEP MIXTE vs ILUT vs ILU(0)
% =========================================================================
addpath('~/chop');
chop([], struct('format', 'h')); % Initialisation environnement half-precision

% ==================== CONFIGURATION GLOBALE ====================
colors = [
    0, 114, 189;    % Mixte (Bleu)
    34, 139, 34;    % ILUT 64 (Vert foncé)
    119, 172, 48;   % ILUT 32 (Vert clair)
    237, 177, 32;   % ILUT 16 (Jaune moutarde)
    162, 20, 47;    % ILU(0) 64 (Rouge foncé)
    217, 83, 25;    % ILU(0) 32 (Rouge orangé)
    126, 47, 142    % ILU(0) 16 (Violet)
] / 255;

solver_names = {'GEP Mixte', 'ILUT FP64', 'ILUT FP32', 'ILUT FP16', 'ILU(0) FP64', 'ILU(0) FP32', 'ILU(0) FP16'};
markers = {'-o', '-^', '-s', '-d', '--^', '--s', '--d'};
linewidths = [3, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
n_solvers = 7;
opt_half = struct('format', 'h');


% =========================================================================
% PARTIE 1 : ANALYSE DE SENSIBILITÉ AU DROPTOL (Sur une matrice type)
% =========================================================================
fprintf('===== PARTIE 1 : ANALYSE DE SENSIBILITÉ (Variation de droptol) =====\n');
load('data/494_bus.mat');
M_sens = Problem.A;
seuils = logspace(-6, -1, 6);
n_seuils = length(seuils);

iters_sens = zeros(n_seuils, n_solvers);
mem_sens = zeros(n_seuils, n_solvers);
fill_in_sens = zeros(n_seuils, n_solvers);
dist_mixte = zeros(n_seuils, 5);

fprintf('%-10s | %5s | %6s | %6s | %6s | %6s | %6s | %6s \n', 'Droptol', 'Mixte', 'ILUT64', 'ILUT32', 'ILUT16', 'ILU064', 'ILU032', 'ILU016');
fprintf('%s\n', repmat('-', 1, 75));

for k = 1:n_seuils
    s = seuils(k);
    
    % --- GEP Mixte ---
    try
        [L_mix, U_mix, P_mix, counts] = gep_mixte(M_sens, s, 1, 1);
        res_mix = test_ilu(M_sens, L_mix, U_mix, P_mix);
        iters_sens(k, 1) = res_mix.iterations;
        mem_sens(k, 1) = (counts(1)*2 + counts(2)*3 + counts(3)*4 + counts(4)*6 + counts(5)*8) / 1024;
        fill_in_sens(k, 1) = (nnz(L_mix) + nnz(U_mix)) / nnz(M_sens);
        dist_mixte(k, :) = counts;
    catch
        iters_sens(k, 1) = 200; mem_sens(k, 1) = Inf; fill_in_sens(k, 1) = NaN;
    end
    
    % --- ILUT ---
    try
        setup_t.type = 'ilutp'; setup_t.droptol = s; setup_t.udiag = 1;
        [L_t, U_t, P_t] = ilu(M_sens, setup_t);
        nnz_t = nnz(L_t) + nnz(U_t) - size(M_sens,1);
        f_t = (nnz(L_t) + nnz(U_t)) / nnz(M_sens);
        fill_in_sens(k, 2:4) = f_t;
        
        res_t64 = test_ilu(M_sens, L_t, U_t, P_t);
        iters_sens(k, 2) = res_t64.iterations; mem_sens(k, 2) = (nnz_t * 8)/1024;
        
        res_t32 = test_ilu(M_sens, spfun(@(x) double(single(x)), L_t), spfun(@(x) double(single(x)), U_t), P_t);
        iters_sens(k, 3) = res_t32.iterations; mem_sens(k, 3) = (nnz_t * 4)/1024;
        
        res_t16 = test_ilu(M_sens, spfun(@(x) chop(full(x), opt_half), L_t), spfun(@(x) chop(full(x), opt_half), U_t), P_t);
        iters_sens(k, 4) = res_t16.iterations; mem_sens(k, 4) = (nnz_t * 2)/1024;
    catch
        iters_sens(k, 2:4) = 200; mem_sens(k, 2:4) = Inf; fill_in_sens(k, 2:4) = NaN;
    end
    
    % --- ILU(0) ---
    try
        setup_0.type = 'nofill'; setup_0.udiag = 1;
        [L_0, U_0] = ilu(M_sens, setup_0);
        nnz_0 = nnz(L_0) + nnz(U_0) - size(M_sens,1);
        f_0 = (nnz(L_0) + nnz(U_0)) / nnz(M_sens);
        fill_in_sens(k, 5:7) = f_0;
        
        res_064 = test_ilu(M_sens, L_0, U_0);
        iters_sens(k, 5) = res_064.iterations; mem_sens(k, 5) = (nnz_0 * 8)/1024;
        
        res_032 = test_ilu(M_sens, spfun(@(x) double(single(x)), L_0), spfun(@(x) double(single(x)), U_0));
        iters_sens(k, 6) = res_032.iterations; mem_sens(k, 6) = (nnz_0 * 4)/1024;
        
        res_016 = test_ilu(M_sens, spfun(@(x) chop(full(x), opt_half), L_0), spfun(@(x) chop(full(x), opt_half), U_0));
        iters_sens(k, 7) = res_016.iterations; mem_sens(k, 7) = (nnz_0 * 2)/1024;
    catch
        iters_sens(k, 5:7) = 200; mem_sens(k, 5:7) = Inf; fill_in_sens(k, 5:7) = NaN;
    end
    
    fprintf('%10.1e | %5d | %6d | %6d | %6d | %6d | %6d | %6d \n', s, iters_sens(k,1), iters_sens(k,2), iters_sens(k,3), iters_sens(k,4), iters_sens(k,5), iters_sens(k,6), iters_sens(k,7));
end

% --- GRAPHIQUE 1 & 2 : Convergence et Fill-in ---
figure('Name', 'Convergence et Fill-in', 'Position', [100 100 1200 400]);

subplot(1,2,1);
i_plot = iters_sens; i_plot(i_plot >= 200 | i_plot == 0) = NaN;
for s_idx = 1:n_solvers
    loglog(seuils, i_plot(:,s_idx), markers{s_idx}, 'LineWidth', 2.5, 'Color', colors(s_idx,:), 'MarkerSize', 8); hold on;
end
set(gca, 'XDir', 'reverse'); grid on;
xlabel('Drop-tolerance (Sévérité du filtre)'); ylabel('Itérations GMRES');
title('Convergence');
legend(solver_names, 'Location', 'best');

subplot(1,2,2);
% On trace seulement Mixte(1), ILUT64(2) et ILU0_64(5) car le fill-in ne change pas selon la précision FP32/FP16
for s_idx = [1, 2, 5]
    semilogx(seuils, fill_in_sens(:,s_idx), markers{s_idx}, 'LineWidth', 2.5, 'Color', colors(s_idx,:), 'MarkerSize', 8); hold on;
end
set(gca, 'XDir', 'reverse'); grid on;
xlabel('Drop-tolerance'); ylabel('Facteur de Fill-in');
title('Évolution du Fill-in');
legend({'GEP Mixte', 'ILUT', 'ILU(0)'}, 'Location', 'best');

% --- GRAPHIQUE 3 : Répartition adaptative des précisions ---
figure('Name', 'Distribution des formats de précision');
dist_pct = dist_mixte ./ sum(dist_mixte, 2) * 100;
b = bar(1:n_seuils, dist_pct, 'stacked', 'FaceColor', 'flat');
b(1).CData = [0.93 0.69 0.13]; % Jaune/Or (16-bits)
b(2).CData = [0.85 0.33 0.10]; % Orange (24-bits)
b(3).CData = [0.30 0.75 0.93]; % Cyan (32-bits)
b(4).CData = [0.47 0.67 0.19]; % Vert (48-bits)
b(5).CData = [0 0.45 0.74];    % Bleu foncé (64-bits)
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.1e', x), seuils, 'UniformOutput', false));
set(gca, 'XTickLabelRotation', 45);
ylabel('Proportion des éléments (%)');
xlabel('Drop-tolerance');
legend('FP16 (Half)', 'FP24', 'FP32 (Single)', 'FP48', 'FP64 (Double)', 'Location', 'bestoutside');
title('Allocation dynamique de la précision matérielle (GEP Mixte)');
grid on;

% --- GRAPHIQUE 4 : Empreinte Mémoire ---
figure('Name', 'Empreinte Mémoire');
m_plot = mem_sens; m_plot(m_plot == Inf) = NaN;
for s_idx = 1:n_solvers
    semilogx(seuils, m_plot(:,s_idx), markers{s_idx}, 'LineWidth', 2.5, 'Color', colors(s_idx,:), 'MarkerSize', 8); hold on;
end
set(gca, 'XDir', 'reverse'); grid on;
xlabel('Drop-tolerance'); ylabel('Taille en Kilo-Octets (KB)');
title('Empreinte Mémoire Théorique du Préconditionneur');
legend(solver_names, 'Location', 'bestoutside');


% =========================================================================
% PARTIE 2 : BENCHMARK MULTI-MATRICES
% =========================================================================
fprintf('\n===== PARTIE 2 : ROBUSTESSE GLOBALE (Profil de Performance) =====\n');
mat_files = dir('data/*.mat');
n_tests = length(mat_files);

iters = zeros(n_tests, n_solvers); 
mem = zeros(n_tests, n_solvers);   
matrices_names = cell(n_tests, 1);
s_fixe = 1e-5; % Droptol fixe

fprintf('Droptol fixé à %.1e pour ILUT et Mixte\n', s_fixe);
fprintf('%-25s | Mixte | ILUT64 | ILUT32 | ILUT16 | ILU064 | ILU032 | ILU016 \n', 'Matrice');
fprintf('%s\n', repmat('-', 1, 95));

for i = 1:n_tests
    matrices_names{i} = strrep(mat_files(i).name, '.mat', '');
    clear Problem A M;
    load(fullfile('data', mat_files(i).name));
    if exist('Problem', 'var'), M = Problem.A;
    elseif exist('A', 'var'), M = A;
    else, continue; 
    end
    
    if size(M,1) ~= size(M,2)
        iters(i, :) = 200; mem(i, :) = Inf;
        fprintf('%-25s | IGNORÉE (non carrée)\n', matrices_names{i});
        continue;
    end
    
    % --- GEP Mixte ---
    try
        [L_mix, U_mix, P_mix, counts] = gep_mixte(M, s_fixe, 1, 1);
        res_mix = test_ilu(M, L_mix, U_mix, P_mix);
        iters(i, 1) = res_mix.iterations;
        mem(i, 1) = (counts(1)*2 + counts(2)*3 + counts(3)*4 + counts(4)*6 + counts(5)*8) / 1024;
    catch
        iters(i, 1) = 200; mem(i, 1) = Inf;
    end
    
    % --- ILUT ---
    try
        setup_t.type = 'ilutp'; setup_t.droptol = s_fixe; setup_t.udiag = 1;
        [L_t, U_t, P_t] = ilu(M, setup_t);
        nnz_t = nnz(L_t) + nnz(U_t) - size(M,1);
        
        res_t64 = test_ilu(M, L_t, U_t, P_t);
        iters(i, 2) = res_t64.iterations; mem(i, 2) = (nnz_t * 8)/1024;
        
        res_t32 = test_ilu(M, spfun(@(x) double(single(x)), L_t), spfun(@(x) double(single(x)), U_t), P_t);
        iters(i, 3) = res_t32.iterations; mem(i, 3) = (nnz_t * 4)/1024;
        
        res_t16 = test_ilu(M, spfun(@(x) chop(full(x), opt_half), L_t), spfun(@(x) chop(full(x), opt_half), U_t), P_t);
        iters(i, 4) = res_t16.iterations; mem(i, 4) = (nnz_t * 2)/1024;
    catch
        iters(i, 2:4) = 200; mem(i, 2:4) = Inf;
    end
    
    % --- ILU(0) ---
    try
        setup_0.type = 'nofill'; setup_0.udiag = 1;
        [L_0, U_0] = ilu(M, setup_0);
        nnz_0 = nnz(L_0) + nnz(U_0) - size(M,1);
        
        res_064 = test_ilu(M, L_0, U_0);
        iters(i, 5) = res_064.iterations; mem(i, 5) = (nnz_0 * 8)/1024;
        
        res_032 = test_ilu(M, spfun(@(x) double(single(x)), L_0), spfun(@(x) double(single(x)), U_0));
        iters(i, 6) = res_032.iterations; mem(i, 6) = (nnz_0 * 4)/1024;
        
        res_016 = test_ilu(M, spfun(@(x) chop(full(x), opt_half), L_0), spfun(@(x) chop(full(x), opt_half), U_0));
        iters(i, 7) = res_016.iterations; mem(i, 7) = (nnz_0 * 2)/1024;
    catch
        iters(i, 5:7) = 200; mem(i, 5:7) = Inf;
    end
    
    fprintf('%-25s | %5d | %6d | %6d | %6d | %6d | %6d | %6d \n', matrices_names{i}(1:min(25,length(matrices_names{i}))), iters(i,1), iters(i,2), iters(i,3), iters(i,4), iters(i,5), iters(i,6), iters(i,7));
end

% -> Tracés Partie 2
% --- Profil de Performance (Dolan-Moré) ---
cost_matrix = mem;
cost_matrix(iters >= 200 | iters == 0) = Inf;
min_cost = min(cost_matrix, [], 2);
min_cost(min_cost == Inf) = 1; 
perf_ratio = cost_matrix ./ min_cost;

tau_max = max(perf_ratio(perf_ratio < Inf));
if isempty(tau_max) || tau_max == 1, tau_max = 5; end
tau_vals = linspace(1, tau_max * 1.1, 1000); 

profile = zeros(length(tau_vals), n_solvers);
for s_idx = 1:n_solvers
    for k = 1:length(tau_vals)
        profile(k, s_idx) = sum(perf_ratio(:, s_idx) <= tau_vals(k)) / n_tests;
    end
end

figure('Name', 'Partie 2 : Profil de Performance', 'Position', [200, 200, 800, 500]);
for s_idx = 1:n_solvers
    plot(tau_vals, profile(:,s_idx), markers{s_idx}(1:end-1), 'LineWidth', linewidths(s_idx), 'Color', colors(s_idx,:)); hold on;
end
grid on;
xlabel('Facteur de surcoût mémoire toléré (\tau)');
ylabel('Fraction des matrices résolues avec succès');
title('Profil de Performance : Efficacité Mémoire');
legend(solver_names, 'Location', 'southeast');
ylim([0 1.05]); xlim([1 max(tau_vals)]);