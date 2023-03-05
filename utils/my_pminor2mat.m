function X = my_pminor2mat(Y, idx,dim,k) 
% Y k^2 x c;
% c is conbination number
c = size(Y,2); 
X = zeros(dim);
for n = 1:c
    temp_idx = idx(n,:);
    temp_minor = reshape(Y(:,n),[k,k]);
    X(temp_idx,temp_idx) = X(temp_idx,temp_idx) + temp_minor;
end
end
