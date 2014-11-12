A = 0 : 119 ;
A = reshape(A, [2, 3, 4, 5]) ;
B = 0 : 2 : 46 ;
B = reshape(B, [2, 4, 3]) ;
tic

AB = contractTensors(A, 4, [2, 3], B, 3, [3, 2]) ;
AB = reshape(AB, [1, 20])

toc