function [Y, Y_w] = vl_myfc(X, W, dzdy)
%[DZDX, DZDF, DZDB] = vl_myconv(X, F, B, DZDY)
%regular fully connected layer
if iscell(X)
    X_t = zeros(size(X{1},1),length(X));
    for ix = 1 : length(X)
        x_t = X{ix};    
        X_t(:,ix) = x_t(:);
    end
    X = X_t;
end


if nargin < 3
    Y = W'*X;
else
    Y = W * dzdy;
    Y_w = X * dzdy';
end