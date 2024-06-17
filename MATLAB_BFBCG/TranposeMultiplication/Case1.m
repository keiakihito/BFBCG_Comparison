mtxAT = single([4.0, 1.0, 1.0;
                             1.0, 3.0, 1.0]) ;

mtxBT = single([1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0]);

mtxB = mtxBT';

mtxC = mtxAT * mtxB;

fprintf("\n~~mtxC~~\n\n");
disp(mtxC);
