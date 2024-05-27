%Let's realize SVD in the way of what I did by hand in MATLAB
% A = UDV'
mtxA = [1, 1; 0, 1; 1, 0];



% (1) Compose V
% Find eigenvectors and eigenvalues for mtxATA,
mtxATA = mtxA' * mtxA;
%disp(mtxATA);

%Compute eigenvalue and eigenvector
[mtxEigVec, mtxEigVal] = eig(mtxATA);

%Eigen vectors are normalized
%disp(mtxEigVec);

%Extract each eigenvector by column
eigVec1 = mtxEigVec(:, 1);
%disp(eigVec1);
eigVec2 = mtxEigVec(:, 2);
%disp(eigVec2);

%Eigen values are diagonal values
%disp(mtxEigVal);

%Extracting eigenvalues from the diagonal matrix
eigVals = diag(mtxEigVal);
lmbd1 = eigVals(1);
%disp(lmbd1);
lmbd2 = eigVals(2);
%disp(lmbd2);

mtxVT = mtxEigVec';
%disp(mtxVT);






%(2) Compose D
%Square root lambda and put diagonal
%Compute each eiganvalues square root and store as vector

%Sort eigen vectors and eigenvalues in descending order and get the sort index
[dscndEigVals, sortIdx] = sort(eigVals, 'descend');
sortedEigVecs = mtxEigVec(:, sortIdx);
sortedEigVals = eigVals(sortIdx);
%disp(sortedEigVals);

%Square root lambdas
sqrtEigVals = sqrt(sortedEigVals);
%disp(sqrtEigVals);

mtxD = diag(sqrtEigVals);
%disp(mtxD);




%(3) Compose U
u1 = mtxA * sortedEigVecs(:, 1);
%disp(u1);
u1Hat = u1 / norm(u1); % Normalize
%disp(u1_hat);

u2 = mtxA * sortedEigVecs(:, 2);
%disp(u2);
u2Hat = u2 / norm(u2); % Normalize

mtxU = [u1Hat, u2Hat];
%disp(mtxU);



%Compare original mtrxA and decomposion ones.
fprintf("\nOriginal matrix A\n");
disp(mtxA);

fprintf("\n\nUDV^{T} matrix A\n");
mtxA_prime = mtxU * mtxD *mtxVT;
disp(mtxA_prime);