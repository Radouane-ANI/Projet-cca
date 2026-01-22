load("data/494_bus.mat")

M = Problem.A;

[L,U] = lu(M);
subplot(1,3,1); spy(M); title('Originale (M)');
subplot(1,3,2); spy(L); title('Facteur L');
subplot(1,3,3); spy(U); title('Facteur U');
