function Y = mysoftmax(X)
% Softmax layer
Xmax = max(X,[],1) ;
s = bsxfun(@minus, X, Xmax);
ex = exp(s) ;
Y = ex./repmat(sum(ex,1),[size(X,1) 1]);
end
