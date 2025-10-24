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

fprintf('D50_train:\n');
fprintf('  Total samples: %d\n', length(labels_train_50));
fprintf('  Class 0 samples: %d\n', sum(labels_train_50 == 0));
fprintf('  Class 1 samples: %d\n\n', sum(labels_train_50 == 1));

fprintf('D500_train:\n');
fprintf('  Total samples: %d\n', length(labels_train_500));
fprintf('  Class 0 samples: %d\n', sum(labels_train_500 == 0));
fprintf('  Class 1 samples: %d\n\n', sum(labels_train_500 == 1));

fprintf('D5000_train:\n');
fprintf('  Total samples: %d\n', length(labels_train_5000));
fprintf('  Class 0 samples: %d\n', sum(labels_train_5000 == 0));
fprintf('  Class 1 samples: %d\n\n', sum(labels_train_5000 == 1));

fprintf('D10K_validate:\n');
fprintf('  Total samples: %d\n', length(labels_validate));
fprintf('  Class 0 samples: %d\n', sum(labels_validate == 0));
fprintf('  Class 1 samples: %d\n\n', sum(labels_validate == 1));

figure('Position', [100, 100, 1200, 800]);

subplot(2, 2, 1);
plot_dataset(X_train_50, labels_train_50, 'D_{train}^{50}');

subplot(2, 2, 2);
plot_dataset(X_train_500, labels_train_500, 'D_{train}^{500}');

subplot(2, 2, 3);
plot_dataset(X_train_5000, labels_train_5000, 'D_{train}^{5000}');

subplot(2, 2, 4);
plot_dataset(X_validate, labels_validate, 'D_{validate}^{10K}');

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

function plot_dataset(X, labels, dataset_name)
    idx_L0 = labels == 0;
    idx_L1 = labels == 1;
    
    plot(X(1, idx_L0), X(2, idx_L0), 'bo', 'MarkerSize', 5, ...
         'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'none');
    hold on;
    
    plot(X(1, idx_L1), X(2, idx_L1), 'rs', 'MarkerSize', 5, ...
         'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none');
    
    xlabel('x_1', 'FontSize', 10);
    ylabel('x_2', 'FontSize', 10);
    title(dataset_name, 'FontSize', 12, 'FontWeight', 'bold');
    legend('Class 0', 'Class 1', 'Location', 'Best', 'FontSize', 9);
    grid on;
    axis equal;
    xlim([-4 4]);
    ylim([-4 4]);
end
