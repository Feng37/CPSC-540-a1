
close all
clear all

% Load data
load outliersData.mat
[n,d] = size(X);

% Plot data
figure(1);
plot(X,y,'b.')
title('Training Data');
hold on

% Fit least-squares estimator
%model = leastSquares(X,y);
model = robustRegression(X,y);
%model = svRegression(X,y,1.0);
% Measure different possible training errors
yhat = model.predict(model,X);
errL1 = sum(abs(y-yhat))/n
errL2 = sum((y-yhat).^2)/n
errLinf = max(abs(y-yhat))

% Draw model prediction
Xsample = [min(X):.01:max(X)]';
yHat = model.predict(model,Xsample);
plot(Xsample,yHat,'g-');
