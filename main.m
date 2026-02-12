load("data/494_bus.mat")

M = Problem.A;
%{
seuils = [0.2, 1e-1, 1e-2, 1e-3, 1e-4];

for s = seuils
    setup.type = 'ilutp';
    setup.droptol = s;
    [L1, U1, P1] = ilu(M, setup);
    %[L2, U2, P2] = gep(M, s);

    res = test_ilu(M,L1,U1);
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
        [L, U] = ilu(M, setup);
        res = test_ilu(M, L, U);
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
%}

[res,s] = opti_dicho(M,0,10,1e-2);
fprintf('\nSeuil: %.3e | Iter: %3d | Fill-in: %.2f | Flag: %d\n', ...
    s, res.iterations, res.fill_in, res.flag);

function [results, seuil] = opti_dicho(A, debut, fin ,precision)
setup.type = 'ilutp';

setup.type = 'ilutp';

while abs(debut - fin) > precision
    mid = (debut + fin)/2;
    fprintf("%f\n",mid);

    setup.droptol = 10^-debut;
    tic
    [Ld, Ud] = ilu(A, setup);
    tpsd = toc;
    resd = test_ilu(A, Ld, Ud);
    Jd = tpsd * resd.fill_in * resd.iterations;
    if resd.flag
        debut = debut + (fin+debut)*0.1;
        continue;
    end

    setup.droptol = 10^-mid;
    tic
    [Lm, Um] = ilu(A, setup);
    tpsm = toc;
    resm = test_ilu(A, Lm, Um);
    Jm = tpsm * resm.fill_in * resm.iterations;

    setup.droptol = 10^-fin;
    tic
    [Lf, Uf] = ilu(A, setup);
    tpsf = toc;
    resf = test_ilu(A, Lf, Uf);
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
