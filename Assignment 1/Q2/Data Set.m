clear; close all; clc; 
rng(12345);  

N = 10000; 
p = [0.25, 0.25, 0.25, 0.25];

MU = [-2, -2; 
       2, -2; 
       2,  2; 
      -2,  2];

S(:,:,1) = [1.0,  0.3;  0.3,  0.8];
S(:,:,2) = [0.7, -0.2; -0.2,  1.2];
S(:,:,3) = [1.1,  0.4;  0.4,  0.9];
S(:,:,4) = [0.8, -0.3; -0.3,  0.7];

lbl = zeros(N, 1);
u = rand(N, 1);

for i = 1:N
    if u(i) < 0.25
        lbl(i) = 1;
    elseif u(i) < 0.50
        lbl(i) = 2;
    elseif u(i) < 0.75
        lbl(i) = 3;
    else
        lbl(i) = 4;
    end
end

X = zeros(N, 2);
for j = 1:4
    idx = (lbl == j);
    X(idx, :) = mvnrnd(MU(j,:), S(:,:,j), sum(idx));
end

fprintf('Samples generated per class:\n');
for j = 1:4
    fprintf('Class %d: %d\n', j, sum(lbl == j));
end
fprintf('Total: %d\n', N);

save Q2_data.mat X lbl MU S p
fprintf('\nData saved to Q2_data.mat\n');
