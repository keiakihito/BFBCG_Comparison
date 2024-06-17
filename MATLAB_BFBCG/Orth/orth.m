format long g

% Set up a sample matrix
% Z = | 1.0  5.0  9.0 |
%     | 2.0  6.0  10.0|
%     | 3.0  7.0  11.0|
%     | 4.0  8.0  12.0|

mtxZ = single([1.0, 5.0, 9.0; 2.0, 6.0, 10.0; 3.0, 7.0, 11.0; 4.0, 8.0, 12.0]);
mtxS = mtxZ' * mtxZ;
fprintf("\n\n~~mtxS~~\n");
disp(mtxS);

threshould = 1e-5;

%Perform SVD
[mtxU, sngVals, mtxVT] = svd(mtxS, 'econ');
fprintf("\n\n~~mtxU~~\n");
disp(mtxU);

fprintf("\n\n~~sngVals~~\n");
disp(sngVals);

fprintf("\n\n~~mtxVT~~\n");
disp(mtxVT);

%Count significan values and set rank, r
r = diag(sngVals) > threshould;
disp(r)

