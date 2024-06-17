%Let's realize SVD in the way of what I did by hand in MATLAB
% A = UDV'
mtxA = [1, 1; 0, 1; 1, 0];



%Compare original mtrxA and decomposion ones.
fprintf("\n\n~~Original matrix A~~\n");
disp(mtxA);

[mtxU, sngVals, mtxVT] = my_SVD_Cmpct(mtxA);

fprintf("\n\n~~matrix U~~\n");
disp(mtxU);

fprintf("\n\n~~Singular Values~~\n");
disp(sngVals);

fprintf("\n\n~~matrix V'~~\n");
disp(mtxVT);

fprintf("\n\n~~UDV^{T} matrix A~~\n");
mtxA_prime = mtxU * sngVals *mtxVT;
disp(mtxA_prime);


fprintf("\n\nCheck solution with svd()\n");
[mtxU_Sol, sngVals_Sol, mtxVT_Sol] = svd(mtxA);

fprintf("\n\n~~matrix U_Sol~~\n");
disp(mtxU);

fprintf("\n\n~~Singular Values_Sol~~\n");
disp(sngVals);

fprintf("\n\n~~matrix V'_Sol~~\n");
disp(mtxVT);


fprintf("\n\nFrom CUDA cusolverDnSgesvd\n");
mtxU_CUDA = [-0.816497, 0.0; -0.408248,  -0.707107;  -0.408248, 0.707107];
mtxD_CUDA = [1.732051, 0 ; 0, 1.0];
mtxVT_CUDA = [-0.707107, -0.707107; 0.707107, -0.707107];

fprintf("\n\n~~matrix U_CUDA~~\n");
disp(mtxU_CUDA);

fprintf("\n\n~~Singular Values_CUDA~~\n");
disp(mtxD_CUDA);

fprintf("\n\n~~matrix V'_CUDA~~\n");
disp(mtxVT_CUDA);



mtxA_Reconstruct = mtxU_CUDA * mtxD_CUDA * mtxVT_CUDA;
fprintf("\n\n~~Reconstruct matrixA with cuda ones~~\n");
disp(mtxA_Reconstruct);


