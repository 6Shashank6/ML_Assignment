clear; close all; clc;

load('question1_datasets.mat');

fprintf('\n========================================\n');
fprintf('Part 2: Logistic Regression Classifiers\n');
fprintf('========================================\n\n');

[w_linear_50, ~] = train_logistic_linear(X_train_50, labels_train_50);
[w_linear_500, ~] = train_logistic_linear(X_train_500, labels_train_500);
[w_linear_5000, ~] = train_logistic_linear(X_train_5000, labels_train_5000);

[w_quad_50, ~] = train_logistic_quadratic(X_train_50, labels_train_50);
[w_quad_500, ~] = train_logistic_quadratic(X_train_500, labels_train_500);
[w_quad_5000, ~] = train_logistic_quadratic(X_train_5000, labels_train_5000);

fprintf('--- Linear: 50 samples ---\n');
[P_error_linear_50, conf_linear_50] = evaluate_classifier(w_linear_50, X_validate, labels_validate, 'linear');
print_results(P_error_linear_50, conf_linear_50);

fprintf('\n--- Linear: 500 samples ---\n');
[P_error_linear_500, conf_linear_500] = evaluate_classifier(w_linear_500, X_validate, labels_validate, 'linear');
print_results(P_error_linear_500, conf_linear_500);

fprintf('\n--- Linear: 5000 samples ---\n');
[P_error_linear_5000, conf_linear_5000] = evaluate_classifier(w_linear_5000, X_validate, labels_validate, 'linear');
print_results(P_error_linear_5000, conf_linear_5000);

fprintf('\n--- Quadratic: 50 samples ---\n');
[P_error_quad_50, conf_quad_50] = evaluate_classifier(w_quad_50, X_validate, labels_validate, 'quadratic');
print_results(P_error_quad_50, conf_quad_50);

fprintf('\n--- Quadratic: 500 samples ---\n');
[P_error_quad_500, conf_quad_500] = evaluate_classifier(w_quad_500, X_validate, labels_validate, 'quadratic');
print_results(P_error_quad_500, conf_quad_500);

fprintf('\n--- Quadratic: 5000 samples ---\n');
[P_error_quad_5000, conf_quad_5000] = evaluate_classifier(w_quad_5000, X_validate, labels_validate, 'quadratic');
print_results(P_error_quad_5000, conf_quad_5000);

fprintf('\n========================================\n');
fprintf('SUMMARY\n');
fprintf('========================================\n\n');
fprintf('Model                   50        500       5000      Optimal\n');
fprintf('---------------------------------------------------------------\n');
fprintf('Logistic-Linear:      %.4f    %.4f    %.4f    %.4f\n', ...
    P_error_linear_50, P_error_linear_500, P_error_linear_5000, P_error_min);
fprintf('Logistic-Quadratic:   %.4f    %.4f    %.4f    %.4f\n', ...
    P_error_quad_50, P_error_quad_500, P_error_quad_5000, P_error_min);

x1_range = linspace(-4, 4, 300);
x2_range = linspace(-4, 4, 300);
[X1_grid, X2_grid] = meshgrid(x1_range, x2_range);

figure('Position', [50, 50, 1400, 900]);

subplot(2, 3, 1);
plot_boundary(X1_grid, X2_grid, w_linear_50, 'linear', X_train_50, labels_train_50, ...
    sprintf('Linear-50\nP(err)=%.4f', P_error_linear_50));

subplot(2, 3, 2);
plot_boundary(X1_grid, X2_grid, w_linear_500, 'linear', X_train_500, labels_train_500, ...
    sprintf('Linear-500\nP(err)=%.4f', P_error_linear_500));

subplot(2, 3, 3);
plot_boundary(X1_grid, X2_grid, w_linear_5000, 'linear', X_train_5000, labels_train_5000, ...
    sprintf('Linear-5000\nP(err)=%.4f', P_error_linear_5000));

subplot(2, 3, 4);
plot_boundary(X1_grid, X2_grid, w_quad_50, 'quadratic', X_train_50, labels_train_50, ...
    sprintf('Quad-50\nP(err)=%.4f', P_error_quad_50));

subplot(2, 3, 5);
plot_boundary(X1_grid, X2_grid, w_quad_500, 'quadratic', X_train_500, labels_train_500, ...
    sprintf('Quad-500\nP(err)=%.4f', P_error_quad_500));

subplot(2, 3, 6);
plot_boundary(X1_grid, X2_grid, w_quad_5000, 'quadratic', X_train_5000, labels_train_5000, ...
    sprintf('Quad-5000\nP(err)=%.4f', P_error_quad_5000));

sgtitle('Decision Boundaries on Training Data', 'FontSize', 14, 'FontWeight', 'bold');

figure('Position', [100, 100, 1400, 900]);

subplot(2, 3, 1);
plot_boundary(X1_grid, X2_grid, w_linear_50, 'linear', X_validate, labels_validate, ...
    sprintf('Linear-50\nP(err)=%.4f', P_error_linear_50));

subplot(2, 3, 2);
plot_boundary(X1_grid, X2_grid, w_linear_500, 'linear', X_validate, labels_validate, ...
    sprintf('Linear-500\nP(err)=%.4f', P_error_linear_500));

subplot(2, 3, 3);
plot_boundary(X1_grid, X2_grid, w_linear_5000, 'linear', X_validate, labels_validate, ...
    sprintf('Linear-5000\nP(err)=%.4f', P_error_linear_5000));

subplot(2, 3, 4);
plot_boundary(X1_grid, X2_grid, w_quad_50, 'quadratic', X_validate, labels_validate, ...
    sprintf('Quad-50\nP(err)=%.4f', P_error_quad_50));

subplot(2, 3, 5);
plot_boundary(X1_grid, X2_grid, w_quad_500, 'quadratic', X_validate, labels_validate, ...
    sprintf('Quad-500\nP(err)=%.4f', P_error_quad_500));

subplot(2, 3, 6);
plot_boundary(X1_grid, X2_grid, w_quad_5000, 'quadratic', X_validate, labels_validate, ...
    sprintf('Quad-5000\nP(err)=%.4f', P_error_quad_5000));

sgtitle('Decision Boundaries on Validation Set (10K samples)', 'FontSize', 14, 'FontWeight', 'bold');

function [w_opt, nll_opt] = train_logistic_linear(X, labels)
    w_init = zeros(3, 1);
    nll_func = @(w) compute_nll(w, X, labels, 'linear');
    options = optimset('Display', 'off', 'MaxIter', 1000, 'MaxFunEvals', 5000);
    [w_opt, nll_opt] = fminsearch(nll_func, w_init, options);
end

function [w_opt, nll_opt] = train_logistic_quadratic(X, labels)
    w_init = zeros(6, 1);
    nll_func = @(w) compute_nll(w, X, labels, 'quadratic');
    options = optimset('Display', 'off', 'MaxIter', 2000, 'MaxFunEvals', 10000);
    [w_opt, nll_opt] = fminsearch(nll_func, w_init, options);
end

function nll = compute_nll(w, X, labels, model_type)
    if strcmp(model_type, 'linear')
        Z = [ones(1, size(X, 2)); X];
    else
        Z = [ones(1, size(X, 2)); X(1,:); X(2,:); ...
             X(1,:).^2; X(1,:).*X(2,:); X(2,:).^2];
    end
    
    scores = w' * Z;
    h = 1 ./ (1 + exp(-scores));
    epsilon = 1e-15;
    h = max(min(h, 1 - epsilon), epsilon);
    nll = -sum(labels' .* log(h) + (1 - labels') .* log(1 - h));
end

function [P_error, confusion] = evaluate_classifier(w, X_validate, labels_validate, model_type)
    N = size(X_validate, 2);
    
    if strcmp(model_type, 'linear')
        Z = [ones(1, N); X_validate];
    else
        Z = [ones(1, N); X_validate(1,:); X_validate(2,:); ...
             X_validate(1,:).^2; X_validate(1,:).*X_validate(2,:); X_validate(2,:).^2];
    end
    
    scores = w' * Z;
    posteriors = 1 ./ (1 + exp(-scores));
    decisions = posteriors > 0.5;
    
    TP = sum(decisions == 1 & labels_validate' == 1);
    TN = sum(decisions == 0 & labels_validate' == 0);
    FP = sum(decisions == 1 & labels_validate' == 0);
    FN = sum(decisions == 0 & labels_validate' == 1);
    
    confusion = [TN, FP; FN, TP];
    P_error = (FP + FN) / N;
end

function print_results(P_error, confusion)
    fprintf('P(error) = %.4f (%.2f%%)\n', P_error, P_error * 100);
    fprintf('\nConfusion Matrix:\n');
    fprintf('                 Predicted L=0   Predicted L=1\n');
    fprintf('True L=0:        %6d          %6d\n', confusion(1,1), confusion(1,2));
    fprintf('True L=1:        %6d          %6d\n', confusion(2,1), confusion(2,2));
end

function plot_boundary(X1_grid, X2_grid, w, model_type, X_data, labels_data, title_str)
    posterior_grid = zeros(size(X1_grid));
    
    for i = 1:numel(X1_grid)
        x = [X1_grid(i); X2_grid(i)];
        
        if strcmp(model_type, 'linear')
            z = [1; x];
        else
            z = [1; x(1); x(2); x(1)^2; x(1)*x(2); x(2)^2];
        end
        
        score = w' * z;
        posterior_grid(i) = 1 / (1 + exp(-score));
    end
    
    contour(X1_grid, X2_grid, posterior_grid, [0.5 0.5], 'k-', 'LineWidth', 2);
    hold on;
    
    if size(X_data, 2) > 1000
        idx = 1:10:size(X_data, 2);
    else
        idx = 1:size(X_data, 2);
    end
    
    idx_L0 = labels_data(idx) == 0;
    idx_L1 = labels_data(idx) == 1;
    
    plot(X_data(1, idx(idx_L0)), X_data(2, idx(idx_L0)), 'bo', ...
         'MarkerSize', 4, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none');
    plot(X_data(1, idx(idx_L1)), X_data(2, idx(idx_L1)), 'rs', ...
         'MarkerSize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none');
    
    xlabel('x_1', 'FontSize', 10);
    ylabel('x_2', 'FontSize', 10);
    title(title_str, 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    axis equal;
    xlim([-4 4]);
    ylim([-4 4]);
    hold off;
end
