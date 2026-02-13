load("data/494_bus.mat")

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
%}

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

resDouble = zeros(1, length(seuils));
resSimple = zeros(1, length(seuils));
i=1;
for s = seuils
    setup.type = 'ilutp';
    setup.droptol = s;
    [L1, U1, P] = ilu(M, setup);
    L2 = single(L1);
    U2 = single(U1);
    res1 = test_ilu(M,L1,U1,P);
    resDouble(i) = res1.iterations;
        warning('off', 'all')
    res2 = test_ilu(M,L2,U2,P);
        warning('on', 'all')

    resSimple(i) = res2.iterations;
    i=i+1;

end
figure; 
clf;
semilogx(seuils, resDouble, 'o-', 'LineWidth', 1.5)
hold on
semilogx(seuils, resSimple, 's-', 'LineWidth', 1.5)
hold off

legend('Préconditionneur double', 'Préconditionneur simple')
xlabel('droptol')
ylabel('Nombre d''itérations')
title('Influence de la précision du préconditionneur')
grid on