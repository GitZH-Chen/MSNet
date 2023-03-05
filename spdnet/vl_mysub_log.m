function [Y] = vl_mysub_log(X, dzdy)
%[DZDX, DZDF, DZDB] = vl_myconv(X, F, B, DZDY)
%regular fully connected layer
batch = length(X);
[k_4,c] = size(X{1});
k_2 = sqrt(k_4);
dim = ((k_2+1)*k_2)/2;
temp_X = cell(batch, 1);
for ix = 1 : batch
    temp = X{ix};
    temp_sub_set = cell(c,1);
    for ic = 1:c
        temp_sub = temp(:,ic);
        temp_sub_set{ic} = reshape(temp_sub,[k_2,k_2]);
    end
    temp_X{ix} = temp_sub_set;
end
if nargin < 2
    Y = zeros(dim*c,batch);
    for ix = 1:batch
        temp_Y = vl_mylog(temp_X{ix});
        temp_sub_log_set = zeros(dim,c);
        for ic = 1:c
            temp_sub_log = temp_Y{ic};
            temp_sub_log_set(:,ic) = temp_sub_log(:);
        end
        Y(:,ix)= temp_sub_log_set(:);
    end
else
    Y = cell(batch,1);
    for ix = 1:batch
        temp_dy = reshape(dzdy(:,ix),[dim,c]);
        temp_gradient = zeros([k_4,c]);
        temp_dY = vl_mylog(temp_X{ix},temp_dy);
        for ic = 1:c
            temp_gradient(:,ic) = temp_dY{ic}(:);
        end
        Y{ix}=temp_gradient;
    end
end