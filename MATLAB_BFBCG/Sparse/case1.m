dnsMtxBT = single([  0.1, 0.2, 0.3, 0.4, 0.5;
                                    0.6, 0.7, 0.8, 0.9, 1.0;
                                    1.1, 1.2, 1.3, 1.4, 1.5]);

dnsMtxI = single([1, 0, 0, 0, 0;
                               0, 1, 0, 0, 0;
                               0, 0, 1, 0, 0;
                               0, 0, 0, 1, 0;
                               0, 0, 0, 0, 1]);

fprintf("\n\n~~mtxI~~\n");
disp(dnsMtxI);

dnsMtxB = dnsMtxBT';

fprintf("\n\n~~mtxB~~\n");
disp(dnsMtxB);


mtxC = dnsMtxI * dnsMtxB;

fprintf("\n\n~~mtxC~~\n");
disp(mtxC);