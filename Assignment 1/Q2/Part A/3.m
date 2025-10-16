clear; close all; clc;
load Q2_data.mat
N = numel(lbl);
K = 4;
post = zeros(N, K);
for j = 1:K
 post(:, j) = mvnpdf(X, MU(j,:), S(:,:,j)) * p(j);
end
[~, D] = max(post, [], 2);
acc = mean(D == lbl);
markers = {'o', 's', '^', 'd'};
marker_names = {'Circle', 'Square', 'Triangle', 'Diamond'};

figure('Color', 'w', 'Position', [100 100 1200 900]);
hold on;
grid on;
axis equal;

for j = 1:4
 idx = (lbl == j);
 correct = idx & (D == j);
 if any(correct)
     plot(X(correct, 1), X(correct, 2), markers{j}, ...
         'Color', [0 0.6 0], 'MarkerSize', 6, 'LineWidth', 1.2);
 end
 incorrect = idx & (D ~= j);
 if any(incorrect)
     plot(X(incorrect, 1), X(incorrect, 2), markers{j}, ...
         'Color', [0.8 0 0], 'MarkerSize', 6, 'LineWidth', 1.2);
 end
end

for j = 1:K
 plot(MU(j,1), MU(j,2), 'kx', 'MarkerSize', 20, 'LineWidth', 3);
 text(MU(j,1)+0.3, MU(j,2)+0.3, sprintf('\\mu_%d', j), ...
     'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
end

xlabel('x_1', 'FontSize', 13);
ylabel('x_2', 'FontSize', 13);
title(sprintf('MAP Classification Results | Accuracy: %.2f%%', 100*acc), ...
    'FontSize', 14, 'FontWeight', 'bold');

h = zeros(1, 6);
for j = 1:4
 h(j) = plot(NaN, NaN, markers{j}, 'Color', [0 0.6 0], ...
     'MarkerSize', 8, 'LineWidth', 1.5);
end
h(5) = plot(NaN, NaN, 'o', 'Color', [0.8 0 0], ...
    'MarkerSize', 8, 'LineWidth', 1.5);
h(6) = plot(NaN, NaN, 'kx', 'MarkerSize', 20, 'LineWidth', 3);

legend_labels = {
 sprintf('Class 1 (%s) - Correct', marker_names{1}), ...
 sprintf('Class 2 (%s) - Correct', marker_names{2}), ...
 sprintf('Class 3 (%s) - Correct', marker_names{3}), ...
 sprintf('Class 4 (%s) - Correct', marker_names{4}), ...
 'Misclassified', ...
 'Class Means'
};
legend(h, legend_labels, 'Location', 'best', 'FontSize', 10);

xlim([min(X(:,1))-1, max(X(:,1))+1]);
ylim([min(X(:,2))-1, max(X(:,2))+1]);

drawnow;  

fprintf('Visualization complete! Accuracy: %.2f%%\n', 100*acc);
