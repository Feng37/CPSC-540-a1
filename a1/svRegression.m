function [model] = svRegression(X,y,epsilon)
[n,d] = size(X);
f = [ones(n,1); zeros(d+1,1)];
A = [-eye(n) ones(n,1) X; -eye(n) -ones(n,1) -X ; -eye(n) zeros(n,d+1)];
b = [epsilon*ones(n,1)+y; epsilon*ones(n,1)-y; zeros(n,1)];
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