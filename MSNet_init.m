function [net] = MSNet_init(varargin)
% MSNet_init initializes a MSNet

rng('default');
rng(0) ;

classNum = 9;
filter = [2,3,4,5];
f = 1/100;
stride = 1;

opts.datadim = [100,80,50,25]; 
opts.layernum = length(opts.datadim)-1;
Winit = cell(opts.layernum,1);
for iw = 1 : opts.layernum % BiMap layers
    A = rand(opts.datadim(iw));
    [U1, ~, ~] = svd(A * A');
    Winit{iw} = U1(:,1:opts.datadim(iw+1));
end

sub_w = cell(length(filter),1);
fc_sub = zeros(length(filter),1);
for ik = 1:length(filter)
    k = filter(ik);
    outsize = (sqrt(opts.datadim(end)) - k)/stride + 1;
    c(ik) = outsize^2;
    dim = filter(ik)^2 * (filter(ik)^2+1)/2;
    fc_sub(ik) = c(ik) * dim;
    A = rand(opts.datadim(end-1));
    [U1, ~, ~] = svd(A * A');
    sub_w{ik} = U1(:,1:opts.datadim(end));
end
fdim = sum(fc_sub);
fc_w{1} = f * randn(fdim, classNum, 'single');

net.layers = {};
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{1}) ;
net.layers{end+1} = struct('type', 'rec') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{2}) ;
net.layers{end+1} = struct('type', 'rec') ; 
for ik = 1:length(filter)
    net.layers{end+1} = struct('type', 'bfc2',...
                          'weight', sub_w{ik}) ;    
    net.layers{end+1} = struct('type', 'rec') ;
    net.layers{end+1} = struct('type', 'subconv',...
                          'k', filter(ik), 'c', c(ik)); 
    net.layers{end+1} = struct('type', 'sub_log') ;
end
net.layers{end+1} = struct('type', 'concat');    
net.layers{end+1} = struct('type', 'fc', 'weight', fc_w{1}); 
net.layers{end+1} = struct('type', 'softmaxloss') ;


