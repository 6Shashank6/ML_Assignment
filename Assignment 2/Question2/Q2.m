clear; close all; clc;

rng(42);

[xTrain, yTrain, xValidate, yValidate] = hw2q2(100, 1000);

Phi_train = compute_features(xTrain);
Phi_val = compute_features(xValidate);

fprintf('========== ML Estimator ==========\n');
w_ML = (Phi_train' * Phi_train) \ (Phi_train' * yTrain');
y_pred_ML = Phi_val * w_ML;
MSE_ML = mean((yValidate' - y_pred_ML).^2);
fprintf('ML Validation MSE: %.6f\n\n', MSE_ML);

sigma2 = mean((yTrain' - Phi_train * w_ML).^2);

fprintf('========== MAP Estimator ==========\n');

gamma_values = logspace(-6, 6, 100);
MSE_MAP_train = zeros(length(gamma_values), 1);
MSE_MAP_val = zeros(length(gamma_values), 1);

for i = 1:length(gamma_values)
    gamma = gamma_values(i);
    lambda = sigma2 / gamma;
    w_MAP = (Phi_train' * Phi_train + lambda * eye(10)) \ (Phi_train' * yTrain');
    
    MSE_MAP_train(i) = mean((yTrain' - Phi_train * w_MAP).^2);
    MSE_MAP_val(i) = mean((yValidate' - Phi_val * w_MAP).^2);
end

[MSE_best, idx_best] = min(MSE_MAP_val);
gamma_best = gamma_values(idx_best);

fprintf('Best gamma: %.2e\n', gamma_best);
fprintf('Best MAP Validation MSE: %.6f\n', MSE_best);

figure('Position', [100, 100, 900, 500]);
semilogx(gamma_values, MSE_MAP_train, 'b-', 'LineWidth', 2);
hold on;
semilogx(gamma_values, MSE_MAP_val, 'r-', 'LineWidth', 2);
semilogx(gamma_best, MSE_best, 'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
yline(MSE_ML, 'y--', 'LineWidth', 2.5);
xlabel('Gamma', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Mean Squared Error', 'FontSize', 12, 'FontWeight', 'bold');
title('Average Squared Error on Validation Dataset vs Gamma', 'FontSize', 14, 'FontWeight', 'bold');
legend('Training MSE', 'Validation MSE', 'Optimal gamma', 'ML', 'Location', 'best');
grid on;

function Phi = compute_features(x)
    N = size(x, 2);
    x1 = x(1, :)';
    x2 = x(2, :)';
    Phi = [ones(N, 1), x1, x2, x1.^2, x1.*x2, x2.^2, x1.^3, x1.^2.*x2, x1.*x2.^2, x2.^3];
end
