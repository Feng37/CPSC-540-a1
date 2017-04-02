function  [model] = softmaxClassifier(X,y)

% Compute sizes
[n,d] = size(X);
k = max(y);

W = zeros(d,k); % Each column is a classifier
for c = 1:k
    yc = ones(n,1); % Treat class 'c' as (+1)
    yc(y ~= c) = -1; % Treat other classes as (-1)
    W(:,c) = findMin(@softmaxloss,W(:,c),500,1,X,yc);
end

model.W = W;
model.predict = @predict;
end

function [yhat] = predict(model,X)
W = model.W;
[~,yhat] = max(X*W,[],2);
end

function [f,g] = softmaxloss(w,X,y)
yXw = y.*(X*w);
f = sum(sum(-X'.*w(:,y))+log(sum(exp(w'*X')))); % Function value
g = -X'*(y./(1+exp(yXw))); % Gradient
end
