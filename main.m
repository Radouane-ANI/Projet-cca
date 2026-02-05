load("data/494_bus.mat")

M = Problem.A;

seuils = [0.2, 1e-1, 1e-2, 1e-3, 1e-4];

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

[res,s] = opti_dicho(M,0.2,1e-15,1e-3);
fprintf('\nSeuil: %.2e | Iter: %3d | Err. Rel: %.2e | Fill-in: %.2f | Flag: %d\n', ...
    s, res.iterations, res.relative_error, res.fill_in, res.flag);


function [results, seuil] = opti_dicho(A, debut, fin ,precision)
setup.type = 'ilutp';

while abs(debut-fin)>precision
    setup.droptol = debut;
    [Ld, Ud] = ilu(A, setup);

    setup.droptol = fin;
    [Lf, Uf] = ilu(A, setup);

    resultsd = test_ilu(A,Ld, Ud);
    resultsf = test_ilu(A,Lf,Uf);

    ratiod = resultsd.fill_in^3*resultsd.iterations;
    ratiof = resultsf.fill_in^3*resultsf.iterations;

    mid = (debut + fin)/2;
    if ratiof < ratiod
        debut = mid;
        results = resultsf;
        seuil = fin;
    else
        fin = mid;
        results = resultsd;
        seuil = debut;
    end
end

end