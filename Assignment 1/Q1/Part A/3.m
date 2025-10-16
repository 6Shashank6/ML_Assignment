clear; close all; clc;
load Q1A_data.mat
logp1=log(mvnpdf(X,mu1.',S1));
logp0=log(mvnpdf(X,mu0.',S0));
scores=logp1-logp0;

lg=unique([linspace(-10,10,1201) log(p0/p1) -Inf Inf]);
y1=(labels==1); y0=(labels==0);
P1=sum(y1); P0=sum(y0);

TPR=zeros(size(lg)); FPR=zeros(size(lg)); FNR=zeros(size(lg)); Perr=zeros(size(lg));
for i=1:numel(lg)
    dec=scores>lg(i);
    TP=sum(dec & y1); FN=sum(~dec & y1);
    FP=sum(dec & y0); TN=sum(~dec & y0);
    TPR(i)=TP/P1; FPR(i)=FP/P0; FNR(i)=FN/P1;
    Perr(i)=FPR(i)*p0 + FNR(i)*p1;
end

[perr_min,idx]=min(Perr);
gamma_hat=exp(lg(idx));
tpr_hat=TPR(idx); fpr_hat=FPR(idx);

figure('Color','w');
plot(FPR,TPR,'LineWidth',2); hold on;
plot([0 1],[0 1],'--','Color',[0.6 0.6 0.6]);
plot(fpr_hat,tpr_hat,'o','MarkerSize',8,'MarkerFaceColor',[0.85 0.1 0.1],'Color',[0.85 0.1 0.1]);
axis([0 1 0 1]); grid on;
xlabel('P(D=1|L=0)'); ylabel('P(D=1|L=1)');
title(sprintf('ROC (ERM/LRT)  |  min P(error)=%.4f  at  \\gamma=%.4f',perr_min,gamma_hat));

theo_gamma=p0/p1;
fprintf('Empirical min-error threshold gamma_hat = %.6f\n',gamma_hat);
fprintf('Minimum empirical P(error) = %.6f\n',perr_min);
fprintf('Point on ROC: FPR=%.6f, TPR=%.6f\n',fpr_hat,tpr_hat);
fprintf('Theoretical gamma (0-1 loss) = %.6f\n',theo_gamma);
fprintf('Ratio gamma_hat/theoretical = %.6f\n',gamma_hat/theo_gamma);
