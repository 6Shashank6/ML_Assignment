clear; close all; clc;
load Q1A_data.mat
logp1=log(mvnpdf(X,mu1.',S1));
logp0=log(mvnpdf(X,mu0.',S0));
scores=logp1-logp0;
lg=[-Inf linspace(-6,6,400) Inf];
TPR=zeros(size(lg)); FPR=zeros(size(lg)); FNR=zeros(size(lg));
y1=(labels==1); y0=(labels==0);
P1=sum(y1); P0=sum(y0);
for i=1:numel(lg)
    dec=scores>lg(i);
    TP=sum(dec & y1);
    FP=sum(dec & y0);
    FN=sum(~dec & y1);
    TPR(i)=TP/P1;
    FPR(i)=FP/P0;
    FNR(i)=FN/P1;
end
figure('Color','w');
plot(FPR,TPR,'LineWidth',2,'Color',[0 0.45 0.74]);
hold on; plot([0 1],[0 1],'--','Color',[0.6 0.6 0.6]);
axis([0 1 0 1]); grid on;
xlabel('P(D=1|L=0)','FontWeight','bold');
ylabel('P(D=1|L=1)','FontWeight','bold');
title('ROC Curve (ERM / LRT)','FontWeight','bold');
