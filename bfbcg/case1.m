format long g

mtxA = single([ 10.840188, 0.394383, 0.000000, 0.000000, 0.000000,
                            0.394383, 10.783099, 0.798440, 0.000000, 0.000000,
                            0.000000, 0.798440, 10.911648, 0.197551, 0.000000,
                            0.000000, 0.000000, 0.197551, 10.335223, 0.768230,
                            0.000000, 0.000000, 0.000000, 0.768230, 10.277775 ]);

mtxSolXT = single([0, 0, 0, 0, 0;
                                0, 0, 0, 0, 0;
                                0, 0, 0, 0, 0]);
mtxSolX = mtxSolXT';

mtxBT = single([1, 1, 1, 1, 1;
                           1, 1, 1, 1, 1;
                           1, 1, 1, 1, 1]);
mtxB = mtxBT';

mtxI =  single([1, 0, 0, 0, 0;
                          0, 1, 0, 0, 0;
                          0, 0, 1, 0, 0;
                          0, 0, 0, 1, 0;
                          0, 0, 0, 0, 1]);
threshold = 1e-5;


% R <- B -AX
mtxR =  mtxB - mtxA * mtxSolX;
fprintf("\n\n~~mtxR~~\n\n");
disp(mtxR);

%Calculate original residual for the relative redsidual during the iteration
orgRsdl = calculateResidual(mtxR);
fprintf("\n\n~~Original residual: %f~~\n\n", orgRsdl);

% Z <- M * R
mtxZ = mtxI * mtxR;
fprintf("\n\n~~mtxZ~~\n\n");
disp(mtxZ);

mtxP = orth(mtxZ, threshold);
fprintf("\n\n~~mtxP~~\n\n");
disp(mtxP);

