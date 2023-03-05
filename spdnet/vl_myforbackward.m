function res = vl_myforbackward(net, x, dzdy, res, varargin)
opts.Dropout = 0;
opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.skipForward = false;
opts.backPropDepth = +inf ;
opts.epsilon = 1e-5;
n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end
drop_out = 1;
num.sub = 1;
num.log = 1;
num.bf2 = 1;
for il = 1:n
    switch net.layers{il}.type
        case 'subconv'
            layers_sub(num.sub ) = il;
            num.sub  = num.sub  + 1;
        case 'sub_log'
            layers_log(num.log) = il;
            num.log = num.log + 1;            
        case 'bfc2'
            layers_bfc2(num.bf2) = il;
            num.bf2 = num.bf2 + 1;                              
        case 'concat'
            layers_concat = il;
    end
end
if opts.Dropout && doder
    temp = rand(size(net.layers{layers_first_sub}.weight,1),1);
    temp(find(temp <= opts.Dropout)) = 0;
    drop_out = temp;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
if ~opts.skipForward
  res(1).x = x ;
end

% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------
layers_first_sub = layers_sub(1);
layers_first_bfc2 = layers_bfc2(1);
count = 0;
for i=1:n
  if opts.skipForward, break; end
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'bfc'
      res(i+1).x = vl_mybfc(res(i).x, l.weight) ; 
    case 'bfc2'
      res(i+1).x = vl_mybfc(res(layers_first_bfc2).x, l.weight) ; 
    case 'rec'
      res(i+1).x = vl_myrec(res(i).x, opts.epsilon) ;  
    case 'log'
      res(i+1).x = vl_mylog(res(i).x) ;       
    case 'subconv'
      res(i+1).x = vl_subconv(res(i).x, l.k,l.c) ;       
    case 'sub_log'
      count = count + 1;
      res(layers_concat).x{count} = vl_mysub_log(res(i).x) ;         
    case 'concat'  
      res(i+1).x = vl_myconcat(res(i).x); 
    case 'relu'
      res(i+1).x = vl_myrelu(res(i).x) ;         
    case 'fc'
      res(i+1).x = vl_myfc(res(i).x, l.weight) ;      
    case 'softmaxloss'
      res(i+1).x = vl_mysoftmaxloss(res(i).x, net.layers{end}.class, 1) ;               
    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  if forget
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

% -------------------------------------------------------------------------
%                                                             Backward pass
% -------------------------------------------------------------------------
temp_X = res(layers_first_bfc2).x;
sub_total_dzdx = cell(length(temp_X),1);
for ix = 1:length(temp_X)
    sub_total_dzdx{ix} = zeros(size(temp_X{ix}));
end
if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:max(1, n-opts.backPropDepth+1)
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'bfc2'
        [res(i).dzdx, res(i).dzdw] = ...
             vl_mybfc(res(layers_first_bfc2).x, l.weight, res(i+1).dzdx) ; 
        sub_total_dzdx = cell_add(sub_total_dzdx, res(i).dzdx);
        if i == layers_first_bfc2
            res(i).dzdx = sub_total_dzdx;
        end   
      case 'bfc'
        [res(i).dzdx, res(i).dzdw] = ...
             vl_mybfc(res(i).x, l.weight, res(i+1).dzdx) ;      
      case 'rec'
        res(i).dzdx = vl_myrec(res(i).x, opts.epsilon, res(i+1).dzdx) ; 
      case 'subconv'
        res(i).dzdx = vl_subconv(res(i).x,l.k,l.c, res(i+1).dzdx) ;
      case 'relu'
        res(i).dzdx = vl_myrelu(res(i).x, res(i+1).dzdx) ;
      case 'sub_log'
         idx = find(i==layers_log);
         temp_dzdy = res(layers_concat).dzdx{idx};
         res(i).dzdx = vl_mysub_log(res(i).x, temp_dzdy) ; 
      case 'log'
         res(i).dzdx = vl_mylog(res(i).x, res(i+1).dzdx) ; 
      case 'concat'  
        res(i).dzdx = vl_myconcat(res(i).x, res(i+1).dzdx);
      case 'fc'
        [res(i).dzdx, res(i).dzdw]  = ...
              vl_myfc(res(i).x, l.weight, res(i+1).dzdx) ;   
      case 'softmaxloss'
        res(i).dzdx = vl_mysoftmaxloss(res(i).x, net.layers{end}.class, 1, 1) ;
    end
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end

