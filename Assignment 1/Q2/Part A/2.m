clear; close all; clc;
load Q2_data.mat

N = numel(lbl);
K = 4;

post = zeros(N, K);
for j = 1:K
    post(:, j) = mvnpdf(X, MU(j,:), S(:,:,j)) * p(j);
end

[~, D] = max(post, [], 2);

C = confusionmat(lbl, D);
Pcond = C ./ sum(C, 2);

acc = mean(D == lbl);

fprintf('\n=== MAP CLASSIFIER RESULTS ===\n');
fprintf('Accuracy: %.4f (%.2f%%)\n\n', acc, 100*acc);
fprintf('Confusion Matrix P(D=i|L=j):\n');
fprintf('       D=1     D=2     D=3     D=4\n');
for i = 1:K
    fprintf('L=%d: ', i);
    fprintf(' %.4f ', Pcond(i,:));
    fprintf('\n');
end
