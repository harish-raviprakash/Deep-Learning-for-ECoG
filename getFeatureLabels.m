clear all;
close all;
clc;

currPath = pwd;
dataPath = strcat('Path\to\file');
addpath(dataPath)
load(strcat(dataPath, '\feature.mat'))
Y = [];
y = A(:,1); % 1st column represents labels
for i = 1:600:size(A,1)
    Y = [Y mode(y(i:i+600-1))];
end
dlmwrite(strcat(dataPath,'\','blockLabels.txt'),Y,'delimiter',' ','newline','pc')