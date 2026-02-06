load("data/494_bus.mat")

M = Problem.A;

seuils = [0.2, 1e-1, 1e-2, 1e-3, 1e-4];
%{
for s = seuils
    setup.type = 'ilutp';
    setup.droptol = s;
    [L1, U1, P1] = ilu(M, setup);
    [L2, U2, P2] = gep(M, s);

    res = test_ilu(M,L2,U2);
    fprintf('Seuil: %.1e | Iter: %3d | Err. Rel: %.2e | Fill-in: %.2f | Flag: %d\n', ...
        s, res.iterations, res.relative_error, res.fill_in, res.flag);

    % On trace la courbe de convergence
    semilogy(res.resvec, 'DisplayName', ['Seuil ' num2str(s)]);
    hold on;
end
legend;
grid on;
title('Influence de la précision du préconditionneur');
%}

[res,s] = opti_dicho(M,0.2,1e-15,1e-3);
fprintf('\nSeuil: %.3e | Iter: %3d | Fill-in: %.2f | Flag: %d\n', ...
    s, res.iterations, res.fill_in, res.flag);

function [results, seuil] = opti_dicho(A, debut, fin ,precision)
setup.type = 'ilutp';

setup.type = 'ilutp';

while abs(debut - fin) > precision

    mid = (debut + fin)/2;

    setup.droptol = debut;
    tic
    [Ld, Ud] = ilu(A, setup);
    tpsd = toc;
    resd = test_ilu(A, Ld, Ud);
    Jd = tpsd * resd.fill_in^3 * resd.iterations;

    setup.droptol = mid;
    tic
    [Lm, Um] = ilu(A, setup);
    tpsm = toc;
    resm = test_ilu(A, Lm, Um);
    Jm = tpsm * resm.fill_in^3 * resm.iterations;

    setup.droptol = fin;
    tic
    [Lf, Uf] = ilu(A, setup);
    tpsf = toc;
    resf = test_ilu(A, Lf, Uf);
    Jf = tpsf * resf.fill_in^3 * resf.iterations;

    if Jm < Jd && Jm < Jf
        debut   = debut + (mid - debut)/2;
        fin     = fin   - (fin - mid)/2;
        results = resm;
        seuil   = mid;

    elseif Jd < Jf
        fin     = mid;
        results = resd;
        seuil   = debut;

    else
        debut   = mid;
        results = resf;
        seuil   = fin;
    end
end
end