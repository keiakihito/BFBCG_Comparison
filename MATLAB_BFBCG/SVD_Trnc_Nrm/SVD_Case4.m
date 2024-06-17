%Let's realize SVD in the way of what I did by hand in MATLAB
% A = UDV'

format long g

mtxA = single([
    1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 1.1;
    0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.0;
    3.1, 4.2, 5.3, 6.4, 7.5, 8.6, 9.7, 0.8, 1.9, 2.0;
    2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 1.1, 2.2;
    1.5, 2.6, 3.7, 4.8, 5.9, 6.0, 7.1, 8.2, 9.3, 0.4;
    0.7, 1.6, 2.5, 3.4, 4.3, 5.2, 6.1, 7.0, 8.9, 9.8;
    2.1, 3.0, 4.9, 5.8, 6.7, 7.6, 8.5, 9.4, 0.3, 1.2;
    3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 0.0, 1.1, 2.2;
    2.5, 3.6, 4.7, 5.8, 6.9, 7.0, 8.1, 9.2, 0.3, 1.4;
    1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1, 0.0;
]);





%Compare original mtrxA and decomposion ones.
fprintf("\n\n~~Original matrix A~~\n");
print_mtx(mtxA);


fprintf("\n\nCheck solution with svd()\n");
[mtxU, mtxD, mtxV] = svd(mtxA);

fprintf("\n\n~~matrix U_Sol~~\n");
print_mtx(mtxU);

fprintf("\n\n~~Singular Values_Sol~~\n");
sngVals = diag(mtxD);
disp(sngVals);

fprintf("\n\n~~matrix V'_Sol~~\n");
disp(mtxV);


%Truncate process
%Dtermin the threashold to determin significant singular values
threashold = 1e-5;

%Determine the rank based on the threashold
rank = sum(sngVals > threashold);

%Truncate the matrix V based on the threashould
mtxV_trnc = mtxV(:, 1:rank);

fprintf("\n\n~~matrix V_trunc~~\n");
disp(mtxV_trnc);

