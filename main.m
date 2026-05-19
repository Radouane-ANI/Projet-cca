if ~exist('chop', 'file')
    addpath('~/chop');
end
chop([], struct('format', 'h'));

% ==================== CONFIGURATION GRAPHIQUE GLOBALE ====================
set(groot, 'defaultAxesFontSize', 12, 'defaultAxesFontName', 'Arial', ...
    'defaultLineLineWidth', 2.5, 'defaultAxesXGrid', 'on', ...
    'defaultAxesYGrid', 'on', 'defaultAxesGridAlpha', 0.15, ...
    'defaultAxesColor', [0.98 0.98 0.98]);

%% 1. PHASE DE DÉCOUVERTE / ANCIENNE APPROCHE
fprintf('===== PHASE 1 : DÉCOUVERTE ET ANALYSE INITIALE =====\n');
data_bcs = load('data/bcsstk08.mat');
if isfield(data_bcs, 'Problem')
    M_bcs = data_bcs.Problem.A;
else
    M_bcs = data_bcs.A;
end

% --- 1.1 Tests préliminaires sur ILU ---
seuils_prelim = [0.2, 1e-1, 1e-2, 1e-3, 1e-4];
figure('Name', 'Convergence Préliminaire ILU', 'Position', [100 100 800 500]);
for s = seuils_prelim
    setup_prelim.type = 'ilutp';
    setup_prelim.droptol = s;
    setup_prelim.udiag = 1;
    try
        [L1, U1, P1] = ilu(M_bcs, setup_prelim);
        res_prelim = test_ilu(M_bcs, L1, U1, P1);
        fprintf('Seuil: %.1e | Iter: %3d | Err. Rel: %.2e | Fill-in: %.2f | Flag: %d\n', ...
            s, res_prelim.iterations, res_prelim.relative_error, res_prelim.fill_in, res_prelim.flag);
        semilogy(res_prelim.resvec, 'DisplayName', ['Seuil ' num2str(s)]);
        hold on;
    catch e
        fprintf('Seuil: %.1e | ÉCHEC: %s\n', s, e.message);
    end
end
legend('Location', 'best');
xlabel('Itérations GMRES');
ylabel('Résidu relatif');
title('Influence de la précision du préconditionneur ILU');
hold off;

% --- 1.2 Courbe de compromis ---
figure('Name', 'Compromis Itérations / Fill-in', 'Position', [150 150 800 500]);
droptols = logspace(-5, 0, 30);
iterations_comp = zeros(length(droptols), 1);
fill_ins_comp = zeros(length(droptols), 1);

for i = 1:length(droptols)
    setup_comp.type = 'ilutp';
    setup_comp.droptol = droptols(i);
    try
        [L_comp, U_comp, P_comp] = ilu(M_bcs, setup_comp);
        res_comp = test_ilu(M_bcs, L_comp, U_comp, P_comp);
        iterations_comp(i) = res_comp.iterations;
        fill_ins_comp(i) = res_comp.fill_in;
    catch
        iterations_comp(i) = NaN;
        fill_ins_comp(i) = NaN;
    end
end

valid_indices = ~isnan(iterations_comp);
droptols_valid = droptols(valid_indices);
iterations_valid = iterations_comp(valid_indices);
fill_ins_valid = fill_ins_comp(valid_indices);

subplot(2,1,1);
semilogx(droptols_valid, iterations_valid, '-ob', 'MarkerFaceColor', 'b');
ylabel("Itérations GMRES"); xlabel('droptol');
title('Compromis entre droptol, itérations et fill-in');
grid on;

subplot(2,1,2);
semilogx(droptols_valid, fill_ins_valid, '-sr', 'MarkerFaceColor', 'r');
ylabel('Fill-in'); xlabel('droptol');
legend('Fill-in', 'Location', 'best');
grid on;

% --- 1.3 Optimisation par dichotomie ---
[res_dicho, s_dicho] = opti_dicho(M_bcs, 0, 10, 1e-2);
fprintf('\nDichotomie - Seuil: %.3e | Iter: %3d | Fill-in: %.2f | Flag: %d\n', ...
    s_dicho, res_dicho.iterations, res_dicho.fill_in, res_dicho.flag);

% --- 1.4 Simulation naïve multi-précision ---
seuils_simul = logspace(-5, -1, 10);
precisions = {'double', 'single', 'fp16', 'bfloat16'};
iters_map = zeros(length(precisions), length(seuils_simul));
rel_err_map = zeros(length(precisions), length(seuils_simul));

fprintf('\nDébut de l''analyse multi-précision simulée...\n');

for i = 1:length(seuils_simul)
    s = seuils_simul(i);
    setup_simul.type = 'ilutp';
    setup_simul.droptol = s;
    setup_simul.udiag = 1;

    [L_simul, U_simul, P_simul] = ilu(M_bcs, setup_simul);

    for j = 1:length(precisions)
        p_type = precisions{j};
        L_mod = simuler_precision(L_simul, p_type);
        U_mod = simuler_precision(U_simul, p_type);
        warning('off', 'all');
        res_simul = test_ilu(M_bcs, L_mod, U_mod, P_simul);
        warning('on', 'all');

        iters_map(j, i) = res_simul.iterations;
        rel_err_map(j, i) = res_simul.relative_error;
    end
end

figure('Name', 'Itérations vs Précision (Simulation)', 'Position', [200 200 800 500]);
couleurs_simul = {'-o', '-s', '-^', '-d'};
for j = 1:length(precisions)
    semilogx(seuils_simul, iters_map(j, :), couleurs_simul{j});
    hold on;
end
set(gca, 'XDir', 'reverse');
legend(precisions, 'Location', 'best', 'Interpreter', 'none');
xlabel('Tolérance de chute (droptol)');
ylabel('Nombre d''itérations de GMRES');
title('Impact de la précision de stockage de ILU sur la convergence');
hold off;

s_cible = 1e-5;
setup_cible.type = 'ilutp';
setup_cible.droptol = s_cible;
[L_cible, U_cible, P_cible] = ilu(M_bcs, setup_cible);

figure('Name', 'Profil de Convergence Simulé', 'Position', [250 250 800 500]);
for j = 1:length(precisions)
    p_type = precisions{j};
    L_mod = simuler_precision(L_cible, p_type);
    U_mod = simuler_precision(U_cible, p_type);
    warning('off', 'all');
    res_cible = test_ilu(M_bcs, L_mod, U_mod, P_cible);
    warning('on', 'all');
    semilogy(0:res_cible.iterations, res_cible.resvec, couleurs_simul{j});
    hold on;
end
legend(precisions, 'Location', 'northeast', 'Interpreter', 'none');
xlabel('Numéro de l''itération GMRES');
ylabel('Résidu Relatif');
title(sprintf('Historique des résidus (droptol = %.1e)', s_cible));
hold off;


%% 2. VERSION FINALE OPTIMISÉE
fprintf('\n===== PHASE 2 : VERSION FINALE OPTIMISÉE (GEP MIXTE vs ILUT) =====\n');

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
linestyles = {'-', '--', ':', '-.', '--', ':', '-.'};
linewidths = [4.0, 2.5, 1.5, 1.2, 2.5, 1.5, 1.2];
n_solvers = 7;
opt_half = struct('format', 'h');

% --- 2.1 Analyse de sensibilité au droptol ---
fprintf('\n--- Analyse de sensibilité (Variation de droptol) ---\n');
data_sens = load('data/494_bus.mat');
if isfield(data_sens, 'Problem')
    M_sens = data_sens.Problem.A;
else
    M_sens = data_sens.A;
end

seuils = logspace(-10, -1, 10);
n_seuils = length(seuils);

iters_sens = zeros(n_seuils, n_solvers);
mem_sens = zeros(n_seuils, n_solvers);
fill_in_sens = zeros(n_seuils, n_solvers);
dist_mixte = zeros(n_seuils, 5);

fprintf('%-10s | %5s | %6s | %6s | %6s | %6s | %6s | %6s \n', 'Droptol', 'Mixte', 'ILUT64', 'ILUT32', 'ILUT16', 'ILU064', 'ILU032', 'ILU016');
fprintf('%s\n', repmat('-', 1, 75));

for k = 1:n_seuils
    s = seuils(k);
    
    % GEP Mixte
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
    
    % ILUT
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
    
    % ILU(0)
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

figure('Name', 'Convergence et Fill-in', 'Position', [300 300 1200 500]);

subplot(1,2,1);
i_plot = iters_sens; 
i_plot(i_plot >= 200 | i_plot == 0) = NaN;
for s_idx = 1:4
    loglog(seuils, i_plot(:,s_idx), markers{s_idx}, 'Color', colors(s_idx,:), ...
           'LineStyle', linestyles{s_idx}, 'LineWidth', linewidths(s_idx), ...
           'MarkerFaceColor', colors(s_idx,:), 'MarkerSize', 11 - 2*s_idx); hold on;
end
set(gca, 'XDir', 'reverse');
xlabel('Tolérance de chute (Droptol)'); ylabel('Itérations GMRES');
title('Convergence selon la tolérance');
legend(solver_names(1:4), 'Location', 'best', 'Interpreter', 'none');

subplot(1,2,2);
for s_idx = [1, 2]
    semilogx(seuils, fill_in_sens(:,s_idx), markers{s_idx}, 'Color', colors(s_idx,:), ...
             'LineStyle', linestyles{s_idx}, 'LineWidth', linewidths(s_idx), ...
             'MarkerFaceColor', colors(s_idx,:), 'MarkerSize', 11 - 2*s_idx); hold on;
end
set(gca, 'XDir', 'reverse');
xlabel('Tolérance de chute (Droptol)'); ylabel('Facteur de Fill-in');
title('Évolution du remplissage');
legend({'GEP Mixte', 'ILUT'}, 'Location', 'best', 'Interpreter', 'none');

figure('Name', 'Distribution des formats de précision', 'Position', [350 350 1100 500]);
row_sums = sum(dist_mixte, 2);
row_sums(row_sums == 0) = 1;
dist_pct = (dist_mixte ./ row_sums) * 100;
b = bar(1:n_seuils, dist_pct, 'stacked');
if ~isempty(b)
    set(b(1), 'FaceColor', [0.93 0.69 0.13]); % FP16
    set(b(2), 'FaceColor', [0.85 0.33 0.10]); % FP24
    set(b(3), 'FaceColor', [0.30 0.75 0.93]); % FP32
    set(b(4), 'FaceColor', [0.47 0.67 0.19]); % FP48
    set(b(5), 'FaceColor', [0 0.45 0.74]);    % FP64
end
set(gca, 'XTick', 1:n_seuils);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.1e', x), seuils, 'UniformOutput', false));
set(gca, 'XTickLabelRotation', 45);
ylabel('Proportion des éléments (%)');
xlabel('Tolérance de chute (Droptol)');
legend({'FP16 (Half)', 'FP24', 'FP32 (Single)', 'FP48', 'FP64 (Double)'}, 'Location', 'bestoutside');
title('Allocation dynamique de la précision matérielle (GEP Mixte)');

figure('Name', 'Empreinte Mémoire', 'Position', [400 400 800 500]);
m_plot = mem_sens; 
m_plot(m_plot == Inf) = NaN;
for s_idx = 1:4
    semilogx(seuils, m_plot(:,s_idx), markers{s_idx}, 'Color', colors(s_idx,:), ...
             'MarkerFaceColor', colors(s_idx,:), 'MarkerSize', 7); hold on;
end
set(gca, 'XDir', 'reverse');
xlabel('Tolérance de chute (Droptol)'); ylabel('Taille estimée (KB)');
title('Empreinte mémoire théorique');
legend(solver_names(1:4), 'Location', 'best', 'Interpreter', 'none');


% --- 2.2 Benchmark multi-matrices ---
fprintf('\n--- Benchmark multi-matrices (Profils de Performance) ---\n');
mat_files = dir('data/*.mat');
n_tests = length(mat_files);

iters = zeros(n_tests, n_solvers); 
mem = zeros(n_tests, n_solvers);   
matrices_names = cell(n_tests, 1);
s_fixe = 1e-5;

fprintf('Droptol fixé à %.1e pour ILUT et Mixte\n', s_fixe);
fprintf('%-25s | Mixte | ILUT64 | ILUT32 | ILUT16 | ILU064 | ILU032 | ILU016 \n', 'Matrice');
fprintf('%s\n', repmat('-', 1, 95));

for i = 1:n_tests
    matrices_names{i} = strrep(mat_files(i).name, '.mat', '');
    
    data_mat = load(fullfile('data', mat_files(i).name));
    if isfield(data_mat, 'Problem')
        M = data_mat.Problem.A;
    elseif isfield(data_mat, 'A')
        M = data_mat.A;
    else
        continue; 
    end
    
    if size(M,1) ~= size(M,2)
        iters(i, :) = 200; mem(i, :) = Inf;
        fprintf('%-25s | IGNORÉE (non carrée)\n', matrices_names{i});
        continue;
    end
    
    % GEP Mixte
    try
        [L_mix, U_mix, P_mix, counts] = gep_mixte(M, s_fixe, 1, 1);
        res_mix = test_ilu(M, L_mix, U_mix, P_mix);
        iters(i, 1) = res_mix.iterations;
        mem(i, 1) = (counts(1)*2 + counts(2)*3 + counts(3)*4 + counts(4)*6 + counts(5)*8) / 1024;
    catch
        iters(i, 1) = 200; mem(i, 1) = Inf;
    end
    
    % ILUT
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
    
    % ILU(0)
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
    
    fprintf('%-25s | %5d | %6d | %6d | %6d | %6d | %6d | %6d \n', ...
        matrices_names{i}(1:min(25,length(matrices_names{i}))), ...
        iters(i,1), iters(i,2), iters(i,3), iters(i,4), iters(i,5), iters(i,6), iters(i,7));
end

% --- Calcul du nombre de matrices résolues pour normaliser les profils ---
n_solved_no_ilu0 = sum(any(iters(:, 1:4) < 200, 2));
if n_solved_no_ilu0 == 0, n_solved_no_ilu0 = n_tests; end

n_solved = sum(any(iters(:, 1:7) < 200, 2));
if n_solved == 0, n_solved = n_tests; end

% --- Profil de Performance Mémoire (SANS ILU0) ---
cost_matrix_no_ilu0 = mem(:, 1:4);
cost_matrix_no_ilu0(iters(:, 1:4) >= 200 | iters(:, 1:4) == 0) = Inf;
min_cost_no_ilu0 = min(cost_matrix_no_ilu0, [], 2);
min_cost_no_ilu0(min_cost_no_ilu0 == 0 | min_cost_no_ilu0 == Inf) = 1e-12; 
perf_ratio_no_ilu0 = cost_matrix_no_ilu0 ./ min_cost_no_ilu0;

tau_max_no_ilu0 = max(perf_ratio_no_ilu0(perf_ratio_no_ilu0 < Inf));
if isempty(tau_max_no_ilu0) || tau_max_no_ilu0 <= 1, tau_max_no_ilu0 = 5; end
tau_vals_no_ilu0 = linspace(1, tau_max_no_ilu0 * 1.1, 1000); 

profile_no_ilu0 = zeros(length(tau_vals_no_ilu0), 4);
for s_idx = 1:4
    for k = 1:length(tau_vals_no_ilu0)
        profile_no_ilu0(k, s_idx) = sum(perf_ratio_no_ilu0(:, s_idx) <= tau_vals_no_ilu0(k)) / n_solved_no_ilu0;
    end
end

figure('Name', 'Profil de Performance Mémoire (SANS ILU0)', 'Position', [420 420 800 550]);
h_lines = zeros(1, 4);
for s_idx = 1:4
    % Ligne continue
    h_lines(s_idx) = plot(tau_vals_no_ilu0, profile_no_ilu0(:,s_idx), 'Color', colors(s_idx,:), ...
         'LineStyle', linestyles{s_idx}, 'LineWidth', linewidths(s_idx)); hold on;
    % Marqueurs clairsemés
    marker_indices = round(linspace(1, length(tau_vals_no_ilu0), 15));
    plot(tau_vals_no_ilu0(marker_indices), profile_no_ilu0(marker_indices, s_idx), ...
         markers{s_idx}(end), 'Color', colors(s_idx,:), 'MarkerFaceColor', colors(s_idx,:), ...
         'MarkerSize', 11 - 2*s_idx, 'LineStyle', 'none');
end
xlabel('Facteur de surcoût mémoire toléré (\tau)');
ylabel('Fraction des matrices résolues avec succès');
title('Profil de Performance : Efficacité Mémoire (SANS ILU0)');
legend(h_lines, solver_names(1:4), 'Location', 'southeast', 'Interpreter', 'none');
ylim([0 1.05]); xlim([1 max(tau_vals_no_ilu0)]);

% --- Profil de Performance Mémoire (AVEC ILU0) ---
cost_matrix = mem;
cost_matrix(iters >= 200 | iters == 0) = Inf;
min_cost = min(cost_matrix, [], 2);
min_cost(min_cost == 0 | min_cost == Inf) = 1e-12; 
perf_ratio = cost_matrix ./ min_cost;

tau_max = max(perf_ratio(perf_ratio < Inf));
if isempty(tau_max) || tau_max <= 1, tau_max = 5; end
tau_vals = linspace(1, tau_max * 1.1, 1000); 

profile = zeros(length(tau_vals), n_solvers);
for s_idx = 1:n_solvers
    for k = 1:length(tau_vals)
        profile(k, s_idx) = sum(perf_ratio(:, s_idx) <= tau_vals(k)) / n_solved;
    end
end

figure('Name', 'Profil de Performance Mémoire', 'Position', [450 450 800 550]);
h_lines = zeros(1, n_solvers);
for s_idx = 1:n_solvers
    % Ligne continue
    h_lines(s_idx) = plot(tau_vals, profile(:,s_idx), 'Color', colors(s_idx,:), ...
         'LineStyle', linestyles{s_idx}, 'LineWidth', linewidths(s_idx)); hold on;
    % Marqueurs clairsemés
    marker_indices = round(linspace(1, length(tau_vals), 15));
    plot(tau_vals(marker_indices), profile(marker_indices, s_idx), ...
         markers{s_idx}(end), 'Color', colors(s_idx,:), 'MarkerFaceColor', colors(s_idx,:), ...
         'MarkerSize', 11 - 2*min(s_idx, 4), 'LineStyle', 'none');
end
xlabel('Facteur de surcoût mémoire toléré (\tau)');
ylabel('Fraction des matrices résolues avec succès');
title('Profil de Performance : Efficacité Mémoire');
legend(h_lines, solver_names, 'Location', 'southeast', 'Interpreter', 'none');
ylim([0 1.05]); xlim([1 max(tau_vals)]);

% --- Profil de Performance Itérations (SANS ILU0) ---
cost_matrix_iters_no_ilu0 = iters(:, 1:4);
cost_matrix_iters_no_ilu0(iters(:, 1:4) >= 200 | iters(:, 1:4) == 0) = Inf;
min_cost_iters_no_ilu0 = min(cost_matrix_iters_no_ilu0, [], 2);
min_cost_iters_no_ilu0(min_cost_iters_no_ilu0 == 0 | min_cost_iters_no_ilu0 == Inf) = 1e-12;
perf_ratio_iters_no_ilu0 = cost_matrix_iters_no_ilu0 ./ min_cost_iters_no_ilu0;

tau_max_iters_no_ilu0 = max(perf_ratio_iters_no_ilu0(perf_ratio_iters_no_ilu0 < Inf));
if isempty(tau_max_iters_no_ilu0) || tau_max_iters_no_ilu0 <= 1, tau_max_iters_no_ilu0 = 5; end
tau_vals_iters_no_ilu0 = linspace(1, tau_max_iters_no_ilu0 * 1.1, 1000);

profile_iters_no_ilu0 = zeros(length(tau_vals_iters_no_ilu0), 4);
for s_idx = 1:4
    for k = 1:length(tau_vals_iters_no_ilu0)
        profile_iters_no_ilu0(k, s_idx) = sum(perf_ratio_iters_no_ilu0(:, s_idx) <= tau_vals_iters_no_ilu0(k)) / n_solved_no_ilu0;
    end
end

figure('Name', 'Profil de Performance Itérations (SANS ILU0)', 'Position', [480 480 800 550]);
h_lines = zeros(1, 4);
for s_idx = 1:4
    % Ligne continue
    h_lines(s_idx) = plot(tau_vals_iters_no_ilu0, profile_iters_no_ilu0(:,s_idx), 'Color', colors(s_idx,:), ...
         'LineStyle', linestyles{s_idx}, 'LineWidth', linewidths(s_idx)); hold on;
    % Marqueurs clairsemés
    marker_indices = round(linspace(1, length(tau_vals_iters_no_ilu0), 15));
    plot(tau_vals_iters_no_ilu0(marker_indices), profile_iters_no_ilu0(marker_indices, s_idx), ...
         markers{s_idx}(end), 'Color', colors(s_idx,:), 'MarkerFaceColor', colors(s_idx,:), ...
         'MarkerSize', 11 - 2*s_idx, 'LineStyle', 'none');
end
xlabel('Facteur de surcoût en itérations toléré (\tau)');
ylabel('Fraction des matrices résolues avec succès');
title('Profil de Performance : Robustesse / Itérations (SANS ILU0)');
legend(h_lines, solver_names(1:4), 'Location', 'southeast', 'Interpreter', 'none');
ylim([0 1.05]); xlim([1 max(tau_vals_iters_no_ilu0)]);

% --- Profil de Performance Itérations (AVEC ILU0) ---
cost_matrix_iters = iters;
cost_matrix_iters(iters >= 200 | iters == 0) = Inf;
min_cost_iters = min(cost_matrix_iters, [], 2);
min_cost_iters(min_cost_iters == 0 | min_cost_iters == Inf) = 1e-12;
perf_ratio_iters = cost_matrix_iters ./ min_cost_iters;

tau_max_iters = max(perf_ratio_iters(perf_ratio_iters < Inf));
if isempty(tau_max_iters) || tau_max_iters <= 1, tau_max_iters = 5; end
tau_vals_iters = linspace(1, tau_max_iters * 1.1, 1000);

profile_iters = zeros(length(tau_vals_iters), n_solvers);
for s_idx = 1:n_solvers
    for k = 1:length(tau_vals_iters)
        profile_iters(k, s_idx) = sum(perf_ratio_iters(:, s_idx) <= tau_vals_iters(k)) / n_solved;
    end
end

figure('Name', 'Profil de Performance Itérations', 'Position', [500 500 800 550]);
h_lines = zeros(1, n_solvers);
for s_idx = 1:n_solvers
    % Ligne continue
    h_lines(s_idx) = plot(tau_vals_iters, profile_iters(:,s_idx), 'Color', colors(s_idx,:), ...
         'LineStyle', linestyles{s_idx}, 'LineWidth', linewidths(s_idx)); hold on;
    % Marqueurs clairsemés
    marker_indices = round(linspace(1, length(tau_vals_iters), 15));
    plot(tau_vals_iters(marker_indices), profile_iters(marker_indices, s_idx), ...
         markers{s_idx}(end), 'Color', colors(s_idx,:), 'MarkerFaceColor', colors(s_idx,:), ...
         'MarkerSize', 11 - 2*min(s_idx, 4), 'LineStyle', 'none');
end
xlabel('Facteur de surcoût en itérations toléré (\tau)');
ylabel('Fraction des matrices résolues avec succès');
title('Profil de Performance : Robustesse / Itérations');
legend(h_lines, solver_names, 'Location', 'southeast', 'Interpreter', 'none');
ylim([0 1.05]); xlim([1 max(tau_vals_iters)]);
