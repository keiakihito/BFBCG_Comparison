%Function to normalize columns of matrix
function nrmdMtx = normalizeClmVec(mtxA)
    [rows, clms] = size(mtxA);
    nrmdMtx = zeros(rows, clms);
    for wkr = 1:clms
        clmVec = mtxA(:, wkr);
        nrmVal = norm(clmVec);
        nrmdMtx(:, wkr) = clmVec / nrmVal;
    end
end
