function Y = vl_mysoftmaxloss(X, c, lambda, dzdy)
% Softmax layer

% class c = 0 skips a spatial location
mass = single(c > 0) ;
mass = mass';

% convert to indexes
c_ = c - 1 ;
for ic = 1  : length(c)
    c_(ic) = c(ic)+(ic-1)*size(X,1);
end

% compute softmaxloss
Xmax = max(X,[],1) ;
ex = exp(bsxfun(@minus, X, Xmax)) ;

if nargin < 4
  t = Xmax + log(sum(ex,1)) - reshape(X(c_), [1 size(X,2)]) ;
  Y = lambda * sum(sum(mass .* t,1)) ;
else
  Y = bsxfun(@rdivide, ex, sum(ex,1)) ;
  Y(c_) = Y(c_) - 1;
  Y = lambda * bsxfun(@times, Y, bsxfun(@times, mass, dzdy)) ;
end