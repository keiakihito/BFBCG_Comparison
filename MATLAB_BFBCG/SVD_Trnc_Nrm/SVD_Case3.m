%Let's realize SVD in the way of what I did by hand in MATLAB
% A = UDV'

format long g

mtxA = single([
    1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7;
    0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6;
    3.0, 4.1, 5.2, 6.3, 7.4, 8.5, 9.6;
    2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8;
    1.4, 2.5, 3.6, 4.7, 5.8, 6.9, 7.0;
    0.6, 1.5, 2.4, 3.3, 4.2, 5.1, 6.0;
    2.0, 2.9, 3.8, 4.7, 5.6, 6.5, 7.4;
]);




%Compare original mtrxA and decomposion ones.
fprintf("\n\n~~Original matrix A~~\n");
disp(mtxA);


fprintf("\n\nCheck solution with svd()\n");
[mtxU, mtxD, mtxV] = svd(mtxA);

fprintf("\n\n~~matrix U_Sol~~\n");
print_mtx(mtxU);

fprintf("\n\n~~Singular Values_Sol~~\n");
sngVals = diag(mtxD);
print_mtx(sngVals);

fprintf("\n\n~~matrix V_Sol~~\n");
disp(mtxV);



%Truncate process
%Dtermin the threashold to determin significant singular values
threashold = 1e-5;

%Determine the rank based on the threashold
rank = sum(sngVals > threashold);
disp(rank);

%Truncate the matrix V based on the threashould
mtxV_trnc = mtxV(:, 1:rank);

fprintf("\n\n~~matrix V_trunc~~\n");
disp(mtxV_trnc);


