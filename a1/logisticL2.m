function [model] = regularizedLogisticRegression(X,y,lambda)

% Add bias variable
[n,d] = size(X);
X = [ones(n,1) X];

% Initial values of regression parameters
w = zeros(d+1,1);

% Solve logistic regression problem
maxIter = 500;
verbose = 1;
w = findMin(@objective,w,maxIter,verbose,X,y,lambda);

model.w = w;
model.predict = @predict;
end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Xhat = [ones(t,1) Xhat];
w = model.w;
yhat = sign(Xhat*w);
end

function [nll,g] = objective(w,X,y,lambda)
%function [nll,g,H] = objective(w,X,y,lambda)%for 3.1.4
yXw = y.*(X*w);

% Function value
nll = sum(log(1+exp(-yXw))) + (lambda/2)*(w'*w);

% Gradient
sigmoid = 1./(1+exp(-yXw));
g = -X'*(y.*(1-sigmoid)) + lambda*w;
%Hessian
%H = X'*(diag(y.^2.*(sigmoid.^2.*exp(-yXw))))*X + lambda*eye(3);%for 3.1.4
end

