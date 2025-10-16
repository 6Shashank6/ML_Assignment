clear; close all; clc;

wine_data = readtable('winequality-white.csv');
X = table2array(wine_data(:, 1:end-1));
y = table2array(wine_data(:, end));
[N, d] = size(X);
classes = unique(y);
K = length(classes);

fprintf('\n========================================\n');
fprintf('WINE QUALITY DATASET - WHITE WINE\n');
fprintf('========================================\n');
fprintf('Samples: %d | Features: %d | Classes: %d\n', N, d, K);

class_priors = zeros(K, 1);
means = zeros(K, d);
covariances = zeros(d, d, K);

fprintf('\nClass Distribution:\n');
fprintf('%-10s %-10s %-12s %-10s\n', 'Quality', 'Count', 'Percentage', 'Lambda');
fprintf('%s\n', repmat('-', 1, 50));

for i = 1:K
    idx = (y == classes(i));
    n_class = sum(idx);
    class_priors(i) = n_class / N;
    
    X_class = X(idx, :);
    means(i, :) = mean(X_class);
    C_sample = cov(X_class);
    
    alpha = 0.01;
    lambda = alpha * trace(C_sample) / rank(C_sample);
    covariances(:, :, i) = C_sample + lambda * eye(d);
    
    fprintf('%-10d %-10d %-12.1f%% %-10.4f\n', ...
        classes(i), n_class, 100*class_priors(i), lambda);
end

posteriors = zeros(N, K);
for i = 1:K
    posteriors(:, i) = mvnpdf(X, means(i, :), covariances(:, :, i)) * class_priors(i);
end
[~, D] = max(posteriors, [], 2);
D_labels = classes(D);

C_mat = confusionmat(y, D_labels);
accuracy = mean(D_labels == y);

col_sums = sum(C_mat, 1);
col_sums(col_sums == 0) = 1;
Pcond = C_mat ./ col_sums;

fprintf('\n========================================\n');
fprintf('RESULTS\n');
fprintf('========================================\n');
fprintf('Accuracy: %.2f%%\n', 100*accuracy);
fprintf('Error: %.2f%%\n\n', 100*(1-accuracy));

fprintf('Confusion Matrix (Counts):\n');
fprintf('%-8s', '');
for j = 1:K, fprintf('L=%-6d', classes(j)); end
fprintf('\n');
for i = 1:K
    fprintf('D=%-6d', classes(i));
    fprintf('%-7d', C_mat(i, :));
    fprintf('\n');
end

fprintf('\nConfusion Matrix P(D=i|L=j):\n');
fprintf('%-8s', '');
for j = 1:K, fprintf('L=%-7d', classes(j)); end
fprintf('\n');
for i = 1:K
    fprintf('D=%-6d', classes(i));
    fprintf('%-8.3f', Pcond(i, :));
    fprintf('\n');
end

fprintf('\nPer-Class Accuracy:\n');
for i = 1:K
    fprintf('  Quality %d: %.1f%%\n', classes(i), 100*Pcond(i,i));
end

[coeff, score, ~, ~, explained] = pca(X);

figure('Color', 'w', 'Position', [100 100 1000 700]);
hold on; grid on;
colors = lines(K);

for i = 1:K
    idx = (y == classes(i));
    scatter(score(idx, 1), score(idx, 2), 30, colors(i, :), 'filled', ...
        'MarkerEdgeAlpha', 0.5, 'MarkerFaceAlpha', 0.6);
end

xlabel(sprintf('PC1 (%.1f%% variance)', explained(1)), 'FontSize', 12);
ylabel(sprintf('PC2 (%.1f%% variance)', explained(2)), 'FontSize', 12);
title('White Wine Quality: PCA 2D', 'FontSize', 14, 'FontWeight', 'bold');
legend(arrayfun(@(x) sprintf('Quality %d', x), classes, 'UniformOutput', false), ...
    'Location', 'best', 'FontSize', 10);

figure('Color', 'w', 'Position', [100 100 1000 700]);
hold on; grid on;

for i = 1:K
    idx = (y == classes(i));
    scatter3(score(idx, 1), score(idx, 2), score(idx, 3), 20, colors(i, :), ...
        'filled', 'MarkerEdgeAlpha', 0.5);
end

xlabel(sprintf('PC1 (%.1f%%)', explained(1)), 'FontSize', 11);
ylabel(sprintf('PC2 (%.1f%%)', explained(2)), 'FontSize', 11);
zlabel(sprintf('PC3 (%.1f%%)', explained(3)), 'FontSize', 11);
title('White Wine Quality: PCA 3D', 'FontSize', 14, 'FontWeight', 'bold');
legend(arrayfun(@(x) sprintf('Quality %d', x), classes, 'UniformOutput', false), ...
    'Location', 'best', 'FontSize', 9);
view(45, 30);

