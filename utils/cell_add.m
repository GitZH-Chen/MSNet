function Z = cell_add(X, Y)
len = length(X);
Z = cell(len, 1);
for ix = 1:len
    Z{ix} = X{ix} + Y{ix};
end

