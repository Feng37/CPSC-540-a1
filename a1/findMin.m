function [w,f] = findMin(funObj,w,maxEvals,verbose,varargin)
% Find local minimizer of differentiable function

% Parameters of the Optimizaton
optTol = 1e-2;
gamma = 1e-4;

% Evaluate the initial function value and gradient
[f,g] = funObj(w,varargin{:});
%[f,g,H] = funObj(w,varargin{:});%for 3.1.4
funEvals = 1;

alpha = 1;
%X = varargin{1}; lambda = varargin{3};L = 1/4*max(eigs(X'*X))+lambda;alpha = 1/L; %for 3.1.3
while 1
    %% Compute search direction
    d = g;
    %d = H\g;%for 3.1.4
    
    %Hv = @(v) Hvfunc(w,v,varargin{:}); d = pcg(Hv,g,optTol);%for 3.2
    %% Line-search to find an acceptable value of alpha
	w_new = w - alpha*d;
	[f_new,g_new] = funObj(w_new,varargin{:});
    %[f_new,g_new,H_new] = funObj(w_new,varargin{:});%for 3.1.4
	funEvals = funEvals+1;
    
    dirDeriv = g'*d;
    while f_new > f - gamma*alpha*dirDeriv
        if verbose
            fprintf('Backtracking...\n');
        end
        alpha = alpha^2*dirDeriv/(2*(f_new - f + alpha*dirDeriv));
        %alpha = alpha / 2.0;%for 3.1.1
        w_new = w - alpha*d;
        [f_new,g_new] = funObj(w_new,varargin{:});
        %[f_new,g_new,H_new] = funObj(w_new,varargin{:});%for 3.1.4
        funEvals = funEvals+1;
    end
    alphaFinal = alpha;

    %% Update step-size for next iteration
    alpha = 1;
    %v = g_new - g; alpha = -alpha * v' * g_new / (v' * v); %for 3.1.2
    %% Sanity check on step-size
    if ~isLegal(alpha) || alpha < 1e-10 || alpha > 1e10
       alpha = 1; 
    end
    
    %% Update parameters/function/gradient
    w = w_new;
    f = f_new;
    g = g_new;
	%H = H_new;%for 3.1.4
    %% Test termination conditions
	optCond = norm(g,'inf');
    if verbose
        fprintf('%6d %15.5e %15.5e %15.5e\n',funEvals,alphaFinal,f,optCond);
    end
	
	if optCond < optTol
        if verbose
            fprintf('Problem solved up to optimality tolerance\n');
        end
		break;
	end
	
	if funEvals >= maxEvals
        if verbose
            fprintf('At maximum number of function evaluations\n');
        end
		break;
	end
end
end

function [legal] = isLegal(v)
legal = sum(any(imag(v(:))))==0 & sum(isnan(v(:)))==0 & sum(isinf(v(:)))==0;
end

function [Hv] = Hvfunc(w,v,X,y,lambda)
yXw = y.*(X*w);
sigmoid = 1./(1+exp(-yXw));
Hv = X'*(diag(sparse(sigmoid.*(1-sigmoid)))*(X*v)) + lambda*v;
end