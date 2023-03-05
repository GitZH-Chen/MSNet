function Y = vl_myconcat(X, dzdy)
num_branch = length(X);
idx_start = zeros(num_branch, 1);
idx_end = zeros(num_branch, 1);
Y = [];
for ib = 1 : num_branch
    idx_start(ib) = size(Y,1)+1;
    Y = [Y;X{ib}];
    idx_end(ib) = size(Y,1);
end
if  nargin == 2  
    clear Y;
    Y = cell(1, num_branch);
    for ib = 1:num_branch
        temp_start = idx_start(ib);
        temp_end = idx_end(ib);
        Y{ib} = dzdy(temp_start:temp_end,:);
    end
end