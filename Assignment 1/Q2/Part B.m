clear; close all; clc;
load Q2_data.mat

N=numel(lbl); K=4;
u=unique(lbl).';
if isequal(u,[0 1 2 3]), lbl=lbl+1; end
if ~isequal(unique(lbl).',[1 2 3 4]), error('Labels must be 1..4'); end

Lambda=[ 0  10  10 100;
         1   0  10 100;
         1   1   0 100;
         1   1   1   0];

L=zeros(N,K);
for j=1:K
    L(:,j)=mvnpdf(X,MU(j,:),S(:,:,j))*p(j);
end

R = L * Lambda.';                 
[~,D]=min(R,[],2);                

avgRisk = mean( Lambda(sub2ind([K K], D, lbl)) );

C = confusionmat(lbl,D,'Order',1:4);
Pcond = C ./ max(sum(C,2),1);

fprintf('\n=== ERM with Unequal Losses (Part B) ===\n');
fprintf('Average empirical risk (min-ERM) = %.6f\n', avgRisk);
disp('Confusion matrix P(D=i|L=j):'); disp(Pcond);
classCounts = sum(C,2).';
fprintf('Class counts (true L=1..4): %s\n', mat2str(classCounts));
