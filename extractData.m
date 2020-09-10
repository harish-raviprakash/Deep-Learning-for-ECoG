clear all;
close all;
clc;

% Set parameters
blocks = 30; % This defines number of seconds in each block
samplingRate = 1200; % Number of samples/second
stride = 600; %Window size for sliding window approach

% Current path
currPath = pwd;
% Data and channel directories
subDir = strcat(currPath,'\SubjectData4');
channelDir = strcat(currPath,'\channels');

addpath(subDir, channelDir)
% Save data path
storeDir=strcat(currPath,'\activeDataNoOverlap');
if ~exist(storeDir,'dir')
    mkdir(storeDir);
end

%% Load each subject's data
files = dir(strcat(subDir,'\*.mat')); 
B1 = [];
A1 = [];
for i = 1:numel(files)
    M = length(files(i).name);
    subName = files(i).name(1:M-4);
    load(files(i).name)
    A2=A';
    % Channel normalization - as used in EEG
    A = zscore(A2,0,2);
    % Load the channel labels for channel-wise classification
    load(strcat(channelDir,'\',subName,'_pos.mat'));
    load(strcat(channelDir,'\',subName,'_neg.mat'));
    A1 = [A1;A(posCh,:)];
    B1 = [B1;A(negCh,:)];
end
createSlidingActiveTime(A1,B1,blocks,samplingRate,stride,storeDir)