load("data/bcsstk08.mat")

M = Problem.A;
seuils = [0.2, 1e-1, 1e-2, 1e-3, 1e-4];
%{

for s = seuils
    setup.type = 'ilutp';
    setup.droptol = s;
    [L1, U1, P1] = ilu(M, setup);
    %[L2, U2, P2] = gep(M, s);

    res = test_ilu(M,L1,U1,P1);
    fprintf('Seuil: %.1e | Iter: %3d | Err. Rel: %.2e | Fill-in: %.2f | Flag: %d\n', ...
        s, res.iterations, res.relative_error, res.fill_in, res.flag);

    % On trace la courbe de convergence
    semilogy(res.resvec, 'DisplayName', ['Seuil ' num2str(s)]);
    hold on;
end
legend;
grid on;
title('Influence de la précision du préconditionneur');
hold off;

% Courbe de compromis
figure; % Nouvelle figure
droptols = logspace(-5, 0, 30); % Plage de valeurs pour droptol
iterations = zeros(length(droptols), 1);
fill_ins = zeros(length(droptols), 1);

for i = 1:length(droptols)
    setup.type = 'ilutp';
    setup.droptol = droptols(i);
    try
        [L, U, P] = ilu(M, setup);
        res = test_ilu(M, L, U, P);
        iterations(i) = res.iterations;
        fill_ins(i) = res.fill_in;
    catch
        iterations(i) = NaN; % Marquer comme invalide si ilu échoue
        fill_ins(i) = NaN;
    end
end

% Filtrer les valeurs NaN où ilu a pu échouer
valid_indices = ~isnan(iterations);
droptols_valid = droptols(valid_indices);
iterations_valid = iterations(valid_indices);
fill_ins_valid = fill_ins(valid_indices);

% Tracer le nombre d'itérations
yyaxis left;
semilogx(droptols_valid, iterations_valid, '-ob');
ylabel("Nombre d'itérations de GMRES");
xlabel('Tolérance de chute (droptol)');
title('Compromis entre droptol, itérations et fill-in');
grid on;

% Tracer le fill-in
yyaxis right;
semilogx(droptols_valid, fill_ins_valid, '-sr');
ylabel('Facteur de remplissage (fill-in)');
legend("Nombre d'itérations", 'Fill-in', 'Location', 'best');
hold off;

[res,s] = opti_dicho(M,0,10,1e-2);
fprintf('\nSeuil: %.3e | Iter: %3d | Fill-in: %.2f | Flag: %d\n', ...
    s, res.iterations, res.fill_in, res.flag);
%}

function [results, seuil] = opti_dicho(A, debut, fin ,precision)
setup.type = 'ilutp';

setup.type = 'ilutp';

while abs(debut - fin) > precision
    mid = (debut + fin)/2;

    setup.droptol = 10^-debut;
    tic
    [Ld, Ud, Pd] = ilu(A, setup);
    tpsd = toc;
    resd = test_ilu(A, Ld, Ud, Pd);
    Jd = tpsd * resd.fill_in * resd.iterations;
    if resd.flag
        debut = debut + (fin+debut)*0.1;
        continue;
    end

    setup.droptol = 10^-mid;
    tic
    [Lm, Um, Pm] = ilu(A, setup);
    tpsm = toc;
    resm = test_ilu(A, Lm, Um, Pm);
    Jm = tpsm * resm.fill_in * resm.iterations;

    setup.droptol = 10^-fin;
    tic
    [Lf, Uf, Pf] = ilu(A, setup);
    tpsf = toc;
    resf = test_ilu(A, Lf, Uf, Pf);
    Jf = tpsf * resf.fill_in * resf.iterations;

    if Jm < Jd && Jm < Jf
        debut   = debut + (mid - debut)/2;
        fin     = fin   - (fin - mid)/2;
        results = resm;
        seuil   = 10^-mid;

    elseif Jd < Jf
        fin     = mid;
        results = resd;
        seuil   = 10^-debut;

    else
        debut   = mid;
        results = resf;
        seuil   = 10^-fin;
    end
end
end

function Ms = simuler_precision(M, type)
% Extrait uniquement les éléments non-nuls pour ne pas remplir la matrice
[i, j, v] = find(M);

switch type
    case 'single'
        % Simule le 32-bit
        v_new = double(single(v));
    case 'fp16'
        % Simule le format IEEE half-precision (16-bit)
        opt.format = 'h';
        v_new = chop(v, opt);
    case 'bfloat16'
        % Simule le format Google bfloat16 (16-bit, grand exposant)
        opt.format = 'b';
        v_new = chop(v, opt);
    case 'double'
        % Garde le 64-bit d'origine
        v_new = v;
    otherwise
        error('Type de précision non reconnu');
end

% Reconstruit la matrice creuse
Ms = sparse(i, j, v_new, size(M,1), size(M,2));
end

function [Ms, taille] = simuler_precision_optimale(M, tolerance)
% Extraction des éléments non nuls
[i, j, v] = find(M);
n = length(v);
if n == 0
    Ms = M; taille = 0; return;
end

% Préparation des vecteurs de test
% On calcule les approximations pour TOUS les éléments d'un coup
v_bf16 = chop(v, struct('format', 'b'));
v_fp16 = chop(v, struct('format', 'h'));
v_single = double(single(v));

% Calcul des erreurs relatives (vectorisé)
% eps est ajouté au dénominateur pour éviter la division par zéro
err_bf16 = abs(v - v_bf16) ./ (abs(v) + eps);
err_fp16 = abs(v - v_fp16) ./ (abs(v) + eps);
err_single = abs(v - v_single) ./ (abs(v) + eps);

% --- Logique de sélection par masques ---

% 1. Masque pour FP16 (Priorité 1)
is_fp16 = (err_fp16 <= tolerance) & ~isinf(v_fp16) & (v_fp16 ~= 0);

% 2. Masque pour BF16 (Priorité 2, si pas FP16)
is_bf16 = ~is_fp16 & (err_bf16 <= tolerance) & ~isinf(v_bf16);

% 3. Masque pour Single (32-bit) (Priorité 3, si ni FP16 ni BF16)
is_single = ~(is_fp16 | is_bf16) & (err_single <= tolerance);

% 4. Masque pour Double (64-bit) (Le reste)
is_double = ~(is_fp16 | is_bf16 | is_single);

% Construction du vecteur final v_new
v_new = zeros(size(v));
v_new(is_fp16) = v_fp16(is_fp16);
v_new(is_bf16) = v_bf16(is_bf16);
v_new(is_single) = v_single(is_single);
v_new(is_double) = v(is_double);

% Calcul de la taille totale (vectorisé)
% On multiplie le nombre d'éléments de chaque type par leur bit-width
taille = sum(is_fp16)*16 + sum(is_bf16)*16 + sum(is_single)*32 + sum(is_double)*64;

% Reconstruction de la matrice creuse
Ms = sparse(i, j, v_new, size(M, 1), size(M, 2));
end


% Initialisation de la fonction chop
chop([], struct('format', 'h')); % Initialise les paramètres internes

seuils = logspace(-5, -1, 10);
precisions = {'double', 'single', 'fp16', 'bfloat16','mixte'};

% Tableaux pour stocker les résultats
iters_map = zeros(length(precisions), length(seuils));
rel_err_map = zeros(length(precisions), length(seuils));

fprintf('Début de l''analyse multi-précision...\n');

for i = 1:length(seuils)
    s = seuils(i);
    setup.type = 'ilutp';
    setup.droptol = s;
    setup.udiag = 1;

    % Factorisation en double précision
    [L, U, P] = ilu(M, setup);

    for j = 1:length(precisions)
        p_type = precisions{j};
        if strcmp(p_type, 'mixte')
            % Utilisation de la fonction avec tolérance adaptative
            [L_mod, tL] = simuler_precision_optimale(L, s);
            [U_mod, tU] = simuler_precision_optimale(U, s);
            fprintf("taille %d, drop %f\n", tL+tU, s);
        else
            % Dégradation de L et U selon la précision choisie
            L_mod = simuler_precision(L, p_type);
            U_mod = simuler_precision(U, p_type);
        end
        % Test avec GMRES
        warning('off', 'all'); % Désactive les alertes de convergence de GMRES
        res = test_ilu(M, L_mod, U_mod, P);
        warning('on', 'all');

        iters_map(j, i) = res.iterations;
        rel_err_map(j, i) = res.relative_error;
    end
end

%% GRAPHIQUE 1 : Robustesse du nombre d'itérations selon la précision
figure('Name', 'Itérations vs Précision');
clf;
couleurs = {'-ok', '-sb', '-^r', '-dg','--pm'};

for j = 1:length(precisions)
    semilogx(seuils, iters_map(j, :), couleurs{j}, 'LineWidth', 1.5, 'MarkerSize', 6);
    hold on;
end

set(gca, 'XDir', 'reverse'); % Inverser l'axe X pour voir la tolérance la plus stricte à gauche
grid on;
legend(precisions, 'Location', 'best');
xlabel('Tolérance de chute (droptol)');
ylabel('Nombre d''itérations de GMRES');
title('Impact de la précision de stockage de ILU sur la convergence');
hold off;

%% GRAPHIQUE 2 : Profil de convergence pour un Seuil Spécifique (ex: droptol = 1e-3)
s_cible = 1e-5;
setup.droptol = s_cible;
[L, U, P] = ilu(M, setup);

figure('Name', 'Profil de Convergence (droptol = 1e-3)');
clf;

for j = 1:length(precisions)
    p_type = precisions{j};
    if strcmp(p_type, 'mixte')
        % Utilisation de la fonction avec tolérance adaptative
        [L_mod, tL] = simuler_precision_optimale(L, s);
        [U_mod, tU] = simuler_precision_optimale(U, s);
    else
        L_mod = simuler_precision(L, p_type);
        U_mod = simuler_precision(U, p_type);
    end
    res = test_ilu(M, L_mod, U_mod, P);
    fprintf("%d , reussi %d \n", s_cible, res.flag);

    semilogy(0:res.iterations, res.resvec, couleurs{j}, 'LineWidth', 1.5);
    hold on;
end

grid on;
legend(precisions, 'Location', 'northeast');
xlabel('Numéro de l''itération GMRES');
ylabel('Résidu Relatif');
title(sprintf('Historique des résidus (droptol = %.1e)', s_cible));
hold off;