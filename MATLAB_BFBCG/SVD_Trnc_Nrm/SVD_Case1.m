% Let's realize SVD in the way of what I did by hand in MATLAB
% A = UDV'
mtxA = single([1, 1; 0, 1; 1, 0]);

% Compare original mtrxA and decomposion ones.
fprintf("\n\n~~Original matrix A~~\n");
disp(mtxA);

fprintf("\n\nCheck solution with svd()\n");
[mtxU, mtxD, mtxV] = svd(mtxA);

fprintf("\n\n~~matrix U_Sol~~\n");
disp(single(mtxU));

fprintf("\n\n~~Singular Values_Sol~~\n");
sngVals = single(diag(mtxD));
disp(sngVals);

fprintf("\n\n~~matrix V_Sol~~\n");
disp(single(mtxV));

% Truncate process
% Determine the threshold to determine significant singular values
threshold = single(1e-6);

% Determine the rank based on the threshold
rank = sum(sngVals > threshold);

% Truncate the matrix V based on the threshold
mtxV_trnc = mtxV(:, 1:rank);

fprintf("\n\n~~matrix V_trunc~~\n");
disp(mtxV_trnc);

mtxY = mtxA * mtxV_trnc;
fprintf("\n\n~~matrix Y~~\n");
disp(mtxY);

mtxY_hat = normalizeClmVec(mtxY);
fprintf("\n\n~~matrix Y_hat~~\n");
disp(mtxY_hat);

% Function to normalize columns of matrix
function nrmdMtx = normalizeClmVec(mtxA)
    [rows, clms] = size(mtxA);
    nrmdMtx = zeros(rows, clms, 'single');
    for wkr = 1:clms
        clmVec = mtxA(:, wkr);
        nrmVal = norm(clmVec);
        nrmdMtx(:, wkr) = clmVec / nrmVal;
    end
end
