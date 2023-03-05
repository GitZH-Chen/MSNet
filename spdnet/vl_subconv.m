function Y = vl_subconv(X, k, c, dzdy)
% submanifold 
idx = cell(size(X,1),1);
Y = cell(size(X,1),1);
for ix = 1  : size(X,1)
    temp = X{ix};
    [minor_p,idx]= my_sliding(temp, 1, k) ;
    if nargin < 4
        Y{ix} = minor_p;      
    end
end
if nargin == 4
    Y = cell(size(X,1),1);
    dim = size(X{1,1},1);
    for ix = 1  : length(X)
        DLDMinor = dzdy{ix};
        Y{ix} = my_pminor2mat(DLDMinor, idx,dim,k^2);
    end
end