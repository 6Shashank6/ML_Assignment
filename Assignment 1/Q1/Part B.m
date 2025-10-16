clear; close all; clc;
load Q1A_data.mat
x=X; y=labels; m0=mu0.'; m1=mu1.'; y1=(y==1); y0=(y==0); P1=sum(y1); P0=sum(y0);
logp1=log(mvnpdf(x,mu1.',S1)); logp0=log(mvnpdf(x,mu0.',S0)); scores_opt=logp1-logp0;
C=eye(3); logp1_nb=log(mvnpdf(x,m1,C)); logp0_nb=log(mvnpdf(x,m0,C)); scores_nb=logp1_nb-logp0_nb;
lg=[-Inf linspace(-40,40,1601) Inf];
TPR_A=zeros(size(lg)); FPR_A=zeros(size(lg)); FNR_A=zeros(size(lg)); Perr_A=zeros(size(lg));
TPR_B=zeros(size(lg)); FPR_B=zeros(size(lg)); FNR_B=zeros(size(lg)); Perr_B=zeros(size(lg));
for i=1:numel(lg)
 dA=scores_opt>lg(i);
 TP=sum(dA & y1); FN=sum(~dA & y1); FP=sum(dA & y0); TN=sum(~dA & y0);
 TPR_A(i)=TP/P1; FPR_A(i)=FP/P0; FNR_A(i)=FN/P1; Perr_A(i)=FPR_A(i)*p0+FNR_A(i)*p1;
 dB=scores_nb>lg(i);
 TP=sum(dB & y1); FN=sum(~dB & y1); FP=sum(dB & y0); TN=sum(~dB & y0);
 TPR_B(i)=TP/P1; FPR_B(i)=FP/P0; FNR_B(i)=FN/P1; Perr_B(i)=FPR_B(i)*p0+FNR_B(i)*p1;
end
[perr_min_A,ia]=min(Perr_A); gamma_hat_A=exp(lg(ia)); tprA=TPR_A(ia); fprA=FPR_A(ia);
[perr_min_B,ib]=min(Perr_B); gamma_hat_B=exp(lg(ib)); tprB=TPR_B(ib); fprB=FPR_B(ib);

figure('Color','w','Position',[100 100 900 720]);
plot(FPR_A,TPR_A,'-','LineWidth',2,'Color',[0 0.45 0.74]); hold on;
plot(FPR_B,TPR_B,'-','LineWidth',2,'Color',[0.85 0.33 0.1]);
plot(fprA,tprA,'s','MarkerSize',9,'MarkerFaceColor',[0 0.45 0.74],'Color',[0 0.45 0.74]);
plot(fprB,tprB,'o','MarkerSize',9,'MarkerFaceColor',[0.85 0.33 0.1],'Color',[0.85 0.33 0.1]);
plot([0 1],[0 1],'k--'); axis([0 1 0 1]); grid on;
xlabel('P(D=1|L=0)'); ylabel('P(D=1|L=1)');
title('ROC: True Model (blue) vs Naive Bayes I-cov (orange)');

theo_gamma=p0/p1;
leg1 = sprintf('True model (P_e=%.4f)', perr_min_A);
leg2 = sprintf('Naive Bayes (P_e=%.4f)', perr_min_B);
leg3 = sprintf('Min P_e (true): \\gamma=%.3f, FPR=%.4f, TPR=%.4f', gamma_hat_A, fprA, tprA);
leg4 = sprintf('Min P_e (NB): \\gamma=%.3f, FPR=%.4f, TPR=%.4f', gamma_hat_B, fprB, tprB);
legend({leg1, leg2, leg3, leg4}, 'Location', 'southeast', 'FontSize', 9);

fprintf('True model: gamma_hat=%.6f min P(error)=%.6f ROC(%.4f,%.4f)\n',gamma_hat_A,perr_min_A,fprA,tprA);
fprintf('Naive Bayes I: gamma_hat=%.6f min P(error)=%.6f ROC(%.4f,%.4f)\n',gamma_hat_B,perr_min_B,fprB,tprB);
fprintf('Increase in P(error) with NB: %.6f (%.2f%%)\n',perr_min_B-perr_min_A,100*(perr_min_B-perr_min_A)/perr_min_A);
fprintf('Theoretical gamma (0-1 loss) = %.6f\n',theo_gamma);
