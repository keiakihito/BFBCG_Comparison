format long g

mtxA = single([4.000000, 1.000000, 1.100000;
                             1.000000,  3.000000, 1.200000;
                             1.000000, 1.000000,  2.300000;
                             2.000000, 5.500000, 1.400000;
                             3.000000, 2.100000, 1.500000 ]) ;
fprintf("\n~~mtxA~~\n\n");
disp(mtxA);

mtxBT = single([1.0, 2.0, 3.0, 2.1, 4.4;
                             4.0, 5.0, 6.0, 1.0, 1.2]);

mtxB = mtxBT';
fprintf("\n~~mtxB~~\n\n");
disp(mtxB);

mtxC = mtxA' * mtxB;

fprintf("\n~~mtxC~~\n\n");
disp(mtxC);
