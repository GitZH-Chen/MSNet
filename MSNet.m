function [net, info] = MSNet(varargin)
clc;
clear;
%set up the path
confPath;
%parameter setting
opts.dataDir = fullfile('./data') ;
opts.imdbPathtrain = fullfile(opts.dataDir, 'sample_for_SPDNet');
opts.batchSize = 30 ; 
opts.test.batchSize = 1;
opts.numEpochs = 500;
opts.gpus = [] ;
opts.learningRate = 0.01*ones(1,opts.numEpochs);
opts.weightDecay = 0.8 ;
opts.continue = 1;
%MSNet initialization
[net] = MSNet_init() ;
%loading metadata 
load(opts.imdbPathtrain) ;
%MSNet training
[net, info] = MSNet_train(net, spd_train, opts);


