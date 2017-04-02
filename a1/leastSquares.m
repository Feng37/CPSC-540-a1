function [model] = leastSquares(X,y)

% Compute sizes
[n,d] = size(X);

% Add bias variable
Z = [ones(n,1) X];

% Solve least squares problem
w = (Z'*Z)\Z'*y;

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);

Zhat = [ones(t,1) Xhat];

yhat = Zhat*model.w;
end