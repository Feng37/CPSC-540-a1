function [model] = robustRegression(X,y)
[n,d] = size(X);
f = [ones(n,1); zeros(d+1,1)];
A = [-eye(n) ones(n,1) X; -eye(n) -ones(n,1) -X];
b = [y; -y];
x = linprog(f,A,b);
w = x(n+1:end);
model.w = w;
model.predict = @predict;
end

function [yhat] = predict(model,Xhat)
[n,d] = size(Xhat);
Zhat = [ones(n,1) Xhat];
yhat = Zhat*model.w;
end