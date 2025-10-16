clear; close all; clc; rng(5644);
N=10000; p0=0.65; p1=0.35;
u=rand(1,N)>=p0; N0=sum(u==0); N1=sum(u==1);
mu0=[-0.5;-0.5;-0.5]; S0=[1,-0.5,0.3;-0.5,1,-0.5;0.3,-0.5,1];
X0=mvnrnd(mu0,S0,N0);
mu1=[1;1;1]; S1=[1,0.3,-0.2;0.3,1,0.3;-0.2,0.3,1];
X1=mvnrnd(mu1,S1,N1);
X=[X0;X1]; labels=[zeros(N0,1);ones(N1,1)];
save Q1A_data.mat X labels p0 p1 mu0 S0 mu1 S1
figure; plot3(X0(:,1),X0(:,2),X0(:,3),'.b'); hold on; plot3(X1(:,1),X1(:,2),X1(:,3),'.r'); axis equal; grid on;
