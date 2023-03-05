function Y = vl_myrec(X, epsilon, dzdy)
% Y = VL_MYREC (X, EPSILON, DZDY)
% ReEig layer
Y = cell(size(X));

for c = 1:size(X,2)
    temp_X = X(:,c);
    Us = cell(length(temp_X),1);
    Ss = cell(length(temp_X),1);
    Vs = cell(length(temp_X),1);
    for ix = 1 : length(temp_X)
        [Us{ix},Ss{ix},Vs{ix}] = svd(temp_X{ix});
    end

    D = size(Ss{1},2);

    if nargin < 3
        for ix = 1:length(temp_X)
            [max_S, ~]=max_eig(Ss{ix},epsilon);
            Y{ix,c} = Us{ix}*max_S*Us{ix}';
        end
    else
        a=12;
        for ix = 1:length(temp_X)
            U = Us{ix}; S = Ss{ix}; V = Vs{ix};

            Dmin = D;

            dLdC = double(dzdy{ix}); dLdC = symmetric(dLdC);

            [max_S, max_I]=max_eig(Ss{ix},epsilon);
            dLdV = 2*dLdC*U*max_S;
            dLdS = (diag(not(max_I)))*U'*dLdC*U;


            K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))');
            K(eye(size(K,1))>0)=0;
            K(find(isinf(K)==1))=0; 

            dzdx = U*(symmetric(K'.*(U'*dLdV))+dDiag(dLdS))*U';

            Y{ix,c} =  dzdx; %warning('no normalization');
        end
    end
end

end
