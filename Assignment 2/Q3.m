clear; close all; clc;
rng(50);
sigma_x = 0.25;
sigma_y = 0.25;
sigma_measurement = 0.3;
angle_true = 2*pi*rand();
radius_true = rand();
x_true = radius_true * cos(angle_true);
y_true = radius_true * sin(angle_true);
fprintf('Vehicle Localization Results:\n');
fprintf('True position: [%.2f, %.2f]\n\n', x_true, y_true);
x_range = linspace(-2, 2, 400);
y_range = linspace(-2, 2, 400);
[X_grid, Y_grid] = meshgrid(x_range, y_range);
J_all = cell(4, 1);
for K = 1:4
 angles = linspace(0, 2*pi, K+1);
 angles = angles(1:K);
 landmarks_x = cos(angles);
 landmarks_y = sin(angles);
 range_measurements = zeros(K, 1);
for i = 1:K
 d_true = hypot(x_true - landmarks_x(i), y_true - landmarks_y(i));
 ri = d_true + sigma_measurement*randn;
while ri < 0
 ri = d_true + sigma_measurement*randn;
end
 range_measurements(i) = ri;
end
 J_grid = zeros(size(X_grid));
for i = 1:K
 d_pred = hypot(X_grid - landmarks_x(i), Y_grid - landmarks_y(i));
 J_grid = J_grid + (range_measurements(i) - d_pred).^2 / (2*sigma_measurement^2);
end
 J_grid = J_grid + X_grid.^2/(2*sigma_x^2) + Y_grid.^2/(2*sigma_y^2);
 [~, idxMin] = min(J_grid(:));
 [iyMin, ixMin] = ind2sub(size(J_grid), idxMin);
 x_map = X_grid(iyMin, ixMin);
 y_map = Y_grid(iyMin, ixMin);
 err = hypot(x_map - x_true, y_map - y_true);
 fprintf('K = %d Landmarks:\n', K);
 fprintf(' Range measurements: ');
 fprintf('%.4f ', range_measurements);
 fprintf('\n');
 fprintf(' MAP estimate: [%.3f, %.3f], Error: %.4f\n\n', x_map, y_map, err);
 J_all{K}.J = J_grid;
 J_all{K}.landmarks_x = landmarks_x;
 J_all{K}.landmarks_y = landmarks_y;
 J_all{K}.x_map = x_map;
 J_all{K}.y_map = y_map;
end
J_min_global = min(cellfun(@(S) min(S.J(:)), J_all));
contour_levels = linspace(J_min_global, J_min_global + 20, 20);
figure('Color','w','Position',[100 100 1100 700]);
t = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
for K = 1:4
 nexttile;
 contour(X_grid, Y_grid, J_all{K}.J, contour_levels, 'LineWidth', 1.2); hold on;
 h1 = plot(x_true, y_true, 'r+', 'MarkerSize', 12, 'LineWidth', 2.2);
 h2 = plot(J_all{K}.x_map, J_all{K}.y_map, 'gx', 'MarkerSize', 10, 'LineWidth', 2);
 h3 = plot(J_all{K}.landmarks_x, J_all{K}.landmarks_y, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'y', 'LineWidth', 1.2);
 axis equal; xlim([-2 2]); ylim([-2 2]); grid on;
 caxis([contour_levels(1) contour_levels(end)]);
 title(sprintf('K = %d Landmarks', K));
 xlabel('x'); ylabel('y');
end
cb = colorbar;
cb.Layout.Tile = 'east';
title(t, 'MAP Objective Function Contours');
lgd = legend([h1, h2, h3], {'True Position', 'MAP Estimate', 'Landmarks'});
lgd.Layout.Tile = 'south';
