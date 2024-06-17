% Define a sparse matric in CSR format
%row = [0, 2, 5, 8, 11, 13];
%col = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
%val = [10, 1, 1, 20, 1, 1, 30, 1, 1, 40, 1, 1, 50];

%MATLAB accepts COO format to make sparse matrix
row = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5];
col = [1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5];
val = [10, 1, 1, 20, 1, 1, 30, 1, 1, 40, 1, 1, 50];

% Create the sparse matric
sprMtxA = sparse(row, col, val);
dnsMtxB = [0.1, 0.6, 1.1; 0.2, 0.7, 1.2; 0.3, 0.8, 1.3; 0.4, 0.9, 1.4; 0.5, 1.0, 1.5];
%disp(dnsMtxB);



%Multipy sparse * dense
Ax_0 = sprMtxA * dnsMtxB;

%Convert to sparse matrix
sprAx_0 = sparse(Ax_0);
fprintf("A * X_0 = \n");
disp(full(sprAx_0));