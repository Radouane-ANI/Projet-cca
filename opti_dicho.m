function [results, seuil] = opti_dicho(A, debut, fin, precision)
    setup.type = 'ilutp';
    setup.udiag = 1; % added to avoid zero pivots
    results = struct('iterations', 0, 'fill_in', 0, 'flag', 1);
    seuil = 1;

    while abs(debut - fin) > precision
        mid = (debut + fin)/2;

        setup.droptol = 10^-debut;
        tic
        try
            [Ld, Ud, Pd] = ilu(A, setup);
            tpsd = toc;
            warning('off', 'all');
            resd = test_ilu(A, Ld, Ud, Pd);
            warning('on', 'all');
            Jd = tpsd + resd.fill_in * resd.iterations;
        catch
            Jd = Inf;
            resd.flag = 1;
        end
        if resd.flag
            debut = debut + (fin+debut)*0.1;
            continue;
        end

        setup.droptol = 10^-mid;
        tic
        try
            [Lm, Um, Pm] = ilu(A, setup);
            tpsm = toc;
            warning('off', 'all');
            resm = test_ilu(A, Lm, Um, Pm);
            warning('on', 'all');
            Jm = tpsm + resm.fill_in * resm.iterations;
        catch
            Jm = Inf;
            resm.flag = 1;
        end

        setup.droptol = 10^-fin;
        tic
        try
            [Lf, Uf, Pf] = ilu(A, setup);
            tpsf = toc;
            warning('off', 'all');
            resf = test_ilu(A, Lf, Uf, Pf);
            warning('on', 'all');
            Jf = tpsf + resf.fill_in * resf.iterations;
        catch
            Jf = Inf;
            resf.flag = 1;
        end

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
