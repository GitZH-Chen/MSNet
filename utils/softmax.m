function Y = softmax(X, dzdy)

% compute softmaxloss
Xmax = max(X,[],1) ;
ex = exp(bsxfun(@minus, X, Xmax)) ;
Y = ex/sum(ex);
if nargin == 2
    Y = diag(Y)-Y*Y';
end