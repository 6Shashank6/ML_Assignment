clear; close all; clc;

if exist('X_train.txt','file')

 Xtr = load('X_train.txt'); ytr = load('y_train.txt');

 Xte = load('X_test.txt'); yte = load('y_test.txt');

elseif isfolder('UCI HAR Dataset')

 Xtr = load(fullfile('UCI HAR Dataset','train','X_train.txt'));

 ytr = load(fullfile('UCI HAR Dataset','train','y_train.txt'));

 Xte = load(fullfile('UCI HAR Dataset','test','X_test.txt'));

 yte = load(fullfile('UCI HAR Dataset','test','y_test.txt'));

else

 error('HAR files not found.');

end

X = [Xtr; Xte];

y = double([ytr; yte]);

X = zscore(double(X));

classes = unique(y); K = numel(classes);

[N,d] = size(X);

alpha = 0.1;

logpost = zeros(N,K);

for k = 1:K

 idx = (y == classes(k));

 mu = mean(X(idx,:),1);

 S = cov(X(idx,:));

 r = rank(S);

 lam = alpha * trace(S) / max(r,1);

 S = S + lam*eye(d);

 L = chol(S,'lower');

 md = sum(((X - mu)/L').^2, 2);

 logdetS = 2*sum(log(diag(L)));

 loglik = -0.5*(d*log(2*pi) + logdetS + md);

 logprior = log(mean(idx));

 logpost(:,k) = loglik + logprior;

end

[~,ix] = max(logpost,[],2);

D = classes(ix);

C = confusionmat(y,D,'Order',classes);

Pcond = C ./ max(sum(C,2),1);

err = mean(D ~= y);

fprintf('HAR | N=%d, d=%d, classes=%s\n', N, d, mat2str(classes.'));

fprintf('Error rate = %.4f (%.2f%%)\n', err, 100*err);

disp('Confusion matrix (rows=true, cols=decision):'); disp(C);

disp('Row-normalized P(D=i|L=j):'); disp(Pcond);

[~,score] = pca(X,'NumComponents',3);

figure('Color','w');

scatter3(score(:,1),score(:,2),score(:,3),8,y,'filled'); grid on;

xlabel('PC1'); ylabel('PC2'); zlabel('PC3');

title(sprintf('HAR PCA(3D) | Err=%.2f%%',100*err));

colorbar;
