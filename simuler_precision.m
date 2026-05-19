function Ms = simuler_precision(M, type)
    [i, j, v] = find(M);

    switch type
        case 'single'
            v_new = double(single(v));
        case 'fp16'
            opt.format = 'h';
            v_new = chop(v, opt);
        case 'bfloat16'
            opt.format = 'b';
            v_new = chop(v, opt);
        case 'double'
            v_new = v;
        otherwise
            error('Type de précision non reconnu');
    end
    Ms = sparse(i, j, v_new, size(M,1), size(M,2));
end
