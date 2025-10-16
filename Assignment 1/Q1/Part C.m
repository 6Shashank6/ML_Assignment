clear; close all; clc;
load Q1A_data.mat
X0=X(labels==0,:); X1=X(labels==1,:);
N0=size(X0,1); N1=size(X1,1);
m0=mean(X0)'; m1=mean(X1)';
S0=cov(X0); S1=cov(X1);
Sw=(S0+S1)/2;
w=Sw\(m1-m0);
w=w/norm(w);
z=X*w;
if mean(z(labels==0))>mean(z(labels==1))
 w=-w; z=-z;
end
tau=linspace(min(z)-1,max(z)+1,1000);
y1=(labels==1); y0=(labels==0);
P1=sum(y1); P0=sum(y0);
TPR_C=zeros(size(tau)); FPR_C=zeros(size(tau)); Pe_C=zeros(size(tau));
for i=1:numel(tau)
 d=z>tau(i);
 TP=sum(d&y1); FP=sum(d&y0);
 TPR_C(i)=TP/P1; FPR_C(i)=FP/P0;
 Pe_C(i)=FPR_C(i)*p0+(1-TPR_C(i))*p1;
end
[Pe_min_C,idx]=min(Pe_C);
fpr_C=FPR_C(idx); tpr_C=TPR_C(idx); tau_opt=tau(idx);
figure('Color','w','Position',[100 100 900 700]);
plot(FPR_C,TPR_C,'LineWidth',2.5,'Color',[0.47 0.67 0.19]); hold on;
plot([0 1],[0 1],'k--','LineWidth',1);
plot(fpr_C,tpr_C,'o','MarkerSize',10,'MarkerFaceColor',[0.47 0.67 0.19],'Color',[0.47 0.67 0.19],'LineWidth',2);
axis([0 1 0 1]); grid on;
xlabel('False Positive Rate P(D=1|L=0)','FontSize',12);
ylabel('True Positive Rate P(D=1|L=1)','FontSize',12);
title(sprintf('Fisher LDA ROC | min P(error)=%.4f',Pe_min_C),'FontSize',13);
leg1 = sprintf('Fisher LDA (P_e=%.4f)', Pe_min_C);
leg2 = 'Random';
leg3 = sprintf('Min P_e: \\tau=%.3f, FPR=%.4f, TPR=%.4f', tau_opt, fpr_C, tpr_C);
legend({leg1, leg2, leg3}, 'Location', 'southeast', 'FontSize',10);

fprintf('\n=== Fisher LDA Results ===\n');
fprintf('Projection vector w_LDA = [%.4f, %.4f, %.4f]^T\n', w(1), w(2), w(3));
fprintf('Optimal threshold tau = %.6f\n',tau_opt);
fprintf('Min P(error) = %.6f\n',Pe_min_C);
fprintf('ROC point (FPR,TPR) = (%.4f, %.4f)\n',fpr_C,tpr_C);
