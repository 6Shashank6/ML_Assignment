clear; close all; clc;

P_L0 = 0.6;
P_L1 = 0.4;

w01 = 0.5; w02 = 0.5;
w11 = 0.5; w12 = 0.5;

m01 = [-0.9; -1.1];
m02 = [0.8; 0.75];
m11 = [-1.1; 0.9];
m12 = [0.9; -0.75];

C = [0.75, 0; 0, 1.25];

rng(42); 

[X_train_50, labels_train_50] = generate_data(50, P_L0, P_L1, ...
    m01, m02, m11, m12, C, w01, w02, w11, w12);

[X_train_500, labels_train_500] = generate_data(500, P_L0, P_L1, ...
    m01, m02, m11, m12, C, w01, w02, w11, w12);

[X_train_5000, labels_train_5000] = generate_data(5000, P_L0, P_L1, ...
    m01, m02, m11, m12, C, w01, w02, w11, w12);

[X_validate, labels_validate] = generate_data(10000, P_L0, P_L1, ...
    m01, m02, m11, m12, C, w01, w02, w11, w12);

fprintf('Datasets generated successfully!\n\n');


N_validate = size(X_validate, 2);
discriminant_scores = zeros(N_validate, 1);

fprintf('Computing discriminant scores...\n');
for i = 1:N_validate
    x = X_validate(:, i);
    
    p_x_given_L0 = w01 * mvnpdf(x', m01', C) + w02 * mvnpdf(x', m02', C);
    p_x_given_L1 = w11 * mvnpdf(x', m11', C) + w12 * mvnpdf(x', m12', C);
    
    posterior_L1 = (p_x_given_L1 * P_L1) / ...
                   (p_x_given_L1 * P_L1 + p_x_given_L0 * P_L0);
    
    discriminant_scores(i) = posterior_L1;
end

decisions_optimal = discriminant_scores > 0.5;

TP_opt = sum(decisions_optimal == 1 & labels_validate == 1);
TN_opt = sum(decisions_optimal == 0 & labels_validate == 0);
FP_opt = sum(decisions_optimal == 1 & labels_validate == 0);
FN_opt = sum(decisions_optimal == 0 & labels_validate == 1);

P_error_min = (FP_opt + FN_opt) / N_validate;

fprintf('\n=== Theoretically Optimal Classifier Results ===\n');
fprintf('Minimum P(error) = %.4f (%.2f%%)\n', P_error_min, P_error_min*100);
fprintf('Correct classifications: %d / %d\n', TP_opt + TN_opt, N_validate);
fprintf('\nConfusion Matrix:\n');
fprintf('                 Predicted L=0   Predicted L=1\n');
fprintf('True L=0:        %6d          %6d\n', TN_opt, FP_opt);
fprintf('True L=1:        %6d          %6d\n', FN_opt, TP_opt);
fprintf('\nTrue Positive Rate (TPR): %.4f\n', TP_opt / (TP_opt + FN_opt));
fprintf('False Positive Rate (FPR): %.4f\n', FP_opt / (FP_opt + TN_opt));


fprintf('\nGenerating ROC curve...\n');

[sorted_scores, sort_idx] = sort(discriminant_scores, 'descend');
sorted_labels = labels_validate(sort_idx);

num_positive = sum(labels_validate == 1);
num_negative = sum(labels_validate == 0);

TPR = zeros(N_validate + 1, 1);
FPR = zeros(N_validate + 1, 1);

TPR(1) = 0;
FPR(1) = 0;

TP = 0;
FP = 0;
for i = 1:N_validate
    if sorted_labels(i) == 1
        TP = TP + 1;
    else
        FP = FP + 1;
    end
    TPR(i+1) = TP / num_positive;
    FPR(i+1) = FP / num_negative;
end

TPR_min_error = TP_opt / num_positive;
FPR_min_error = FP_opt / num_negative;

figure('Position', [100, 100, 800, 700]);
plot(FPR, TPR, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1.5); 
plot(FPR_min_error, TPR_min_error, 'ro', 'MarkerSize', 14, ...
     'MarkerFaceColor', 'r', 'LineWidth', 2.5); 
grid on;
xlabel('False Positive Rate (FPR)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('True Positive Rate (TPR)', 'FontSize', 13, 'FontWeight', 'bold');
title('ROC Curve - Theoretically Optimal Classifier', 'FontSize', 15, 'FontWeight', 'bold');
legend('ROC Curve', 'Random Classifier', ...
       sprintf('Min-P(error) Point (P_{error}=%.4f)', P_error_min), ...
       'Location', 'SouthEast', 'FontSize', 11);
axis([0 1 0 1]);
set(gca, 'FontSize', 12);

text(FPR_min_error + 0.05, TPR_min_error - 0.05, ...
     sprintf('(%.3f, %.3f)', FPR_min_error, TPR_min_error), ...
     'FontSize', 11, 'FontWeight', 'bold');

fprintf('Generating decision boundary plot...\n');

figure('Position', [150, 150, 900, 700]);

x1_range = linspace(-4, 4, 400);
x2_range = linspace(-4, 4, 400);
[X1_grid, X2_grid] = meshgrid(x1_range, x2_range);

posterior_grid = zeros(size(X1_grid));
for i = 1:numel(X1_grid)
    x = [X1_grid(i); X2_grid(i)];
    
    p_x_given_L0 = w01 * mvnpdf(x', m01', C) + w02 * mvnpdf(x', m02', C);
    p_x_given_L1 = w11 * mvnpdf(x', m11', C) + w12 * mvnpdf(x', m12', C);
    
    posterior_grid(i) = (p_x_given_L1 * P_L1) / ...
                        (p_x_given_L1 * P_L1 + p_x_given_L0 * P_L0);
end

contour(X1_grid, X2_grid, posterior_grid, [0.5 0.5], 'k-', 'LineWidth', 3);
hold on;

subsample_idx = 1:10:N_validate; 
idx_L0 = labels_validate(subsample_idx) == 0;
idx_L1 = labels_validate(subsample_idx) == 1;

plot(X_validate(1, subsample_idx(idx_L0)), X_validate(2, subsample_idx(idx_L0)), ...
     'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none');
plot(X_validate(1, subsample_idx(idx_L1)), X_validate(2, subsample_idx(idx_L1)), ...
     'rs', 'MarkerSize', 5, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none');

plot(m01(1), m01(2), 'b^', 'MarkerSize', 15, 'MarkerFaceColor', 'cyan', ...
     'MarkerEdgeColor', 'b', 'LineWidth', 2);
plot(m02(1), m02(2), 'b^', 'MarkerSize', 15, 'MarkerFaceColor', 'cyan', ...
     'MarkerEdgeColor', 'b', 'LineWidth', 2);
plot(m11(1), m11(2), 'rv', 'MarkerSize', 15, 'MarkerFaceColor', 'yellow', ...
     'MarkerEdgeColor', 'r', 'LineWidth', 2);
plot(m12(1), m12(2), 'rv', 'MarkerSize', 15, 'MarkerFaceColor', 'yellow', ...
     'MarkerEdgeColor', 'r', 'LineWidth', 2);

xlabel('x_1', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('x_2', 'FontSize', 13, 'FontWeight', 'bold');
title('Decision Boundary - Theoretically Optimal Classifier', ...
      'FontSize', 15, 'FontWeight', 'bold');
legend('Decision Boundary (P(L=1|x)=0.5)', 'Class 0 samples', 'Class 1 samples', ...
       'Location', 'Best', 'FontSize', 10);
grid on;
axis equal;
xlim([-4 4]);
ylim([-4 4]);
set(gca, 'FontSize', 12);


save('question1_datasets.mat', 'X_train_50', 'labels_train_50', ...
     'X_train_500', 'labels_train_500', 'X_train_5000', 'labels_train_5000', ...
     'X_validate', 'labels_validate', 'P_L0', 'P_L1', ...
     'm01', 'm02', 'm11', 'm12', 'C', 'w01', 'w02', 'w11', 'w12', ...
     'P_error_min');  


function [X, labels] = generate_data(N, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12)
    
    
    X = zeros(2, N);
    labels = zeros(N, 1);
    
    for i = 1:N
        
        if rand() < P_L0
            
            labels(i) = 0;
            
            
            if rand() < w01
                
                X(:, i) = mvnrnd(m01, C)';
            else
                
                X(:, i) = mvnrnd(m02, C)';
            end
        else
            
            labels(i) = 1;
            
            
            if rand() < w11
                
                X(:, i) = mvnrnd(m11, C)';
            else
                
                X(:, i) = mvnrnd(m12, C)';
            end
        end
    end
end
