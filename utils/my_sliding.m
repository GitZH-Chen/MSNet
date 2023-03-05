function [Y,idx] = my_sliding(X, stride, k) 
% X cov matrix
% k filter size
imagesize = sqrt(size(X,1));
outsize = (imagesize - k)/stride + 1;
if imagesize ~= fix(imagesize) && outsize ~= fix(outsize)
    error('wrong size of X')
end
Y = zeros(k^4, outsize^2);
idx = zeros(outsize^2, k^2);
n=1;
for ir = 1:stride:(imagesize-k+1)
    for ic = 1:stride:(imagesize-k+1)
        temp_idx = zeros(k,k);
            for ii = 1:k
                col_start = imagesize*(ic-1) + ir + imagesize * (ii-1);
                col = linspace(col_start,col_start + k - 1, k)';
                temp_idx(:,ii) = col; 
            end
            temp = temp_idx(:)';
            idx(n,:) = temp;
            temp_Y = X(temp,temp);
            Y(:,n) = temp_Y(:);
            n = n+1;
    end
end
end
