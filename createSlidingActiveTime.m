% %  Function to perform data augmentaiton via sliding window
function createSlidingActiveTime(A1,A2,block,samplingRate,stride,savePath)
    blockDim = block*samplingRate; % Number of samples in each block [control/task]
    A = [A1;A2];
    % Generating indices for balanced negative channel sampling
    ind = randperm(size(A2,1));
    % Selecting equal number of positive response and negative response channels
    A = [A1;A2(ind(1:size(A1,1)),:)];
    y = zeros(size(A,1),1);
    y(1:size(A1,1)) = 1;
    clear t1 A1 t2 A2
    % Save indices to re-use
    save(strcat(savePath,'/','slidingWindowApproachInd.mat'),'ind')
    [avg,peak,skew,kurt,activity,mobility,complexity,ar,psd] = partitionData(A,y,blockDim,stride,ARorder,samplingRate);
    A = avg;
    save(strcat(savePath,'/','Mean.mat'),'A')
    A = peak;
    save(strcat(savePath,'/','P2P.mat'),'A')
    A = skew;
    save(strcat(savePath,'/','Skew.mat'),'A')
    A = kurt;
    save(strcat(savePath,'/','Kurtosis.mat'),'A')
    A = activity;
    save(strcat(savePath,'/','Activity.mat'),'A')
    A = mobility;
    save(strcat(savePath,'/','Mobility.mat'),'A')
    A = complexity;
    save(strcat(savePath,'/','Complexity.mat'),'A')
    A = ar;
    save(strcat(savePath,'/','AR.mat'),'A')
    A = psd;
    save(strcat(savePath,'/','PSD.mat'),'A')
end

%Split data into sub-blocks
function [B,E,F,G,H,I,J,C,D] = partitionData(A,y,blockDim,stride,ARorder,SR)
    sampleSize = SR/2; % Window size
    B = []; C = []; D = []; E = []; F = []; G = []; H = []; I = []; J = [];
    [m,n] = size(A);
    for i = 1:m
        tmp = [];
        for j = 1:blockDim:n
            tmp = [tmp;j];
        end
        A1 = A(i,:);
        for j = 1:size(tmp,1)
            t = tmp(j);
            if mod((t-1)/blockDim,2) == 0
                continue;
            else
                b = A1(1,t:t+blockDim-1);
                d = arburg(b,ARorder); % AR features
                e = pcov(b,ARorder,SR);
                tmp1 = []; tmp2 = []; tmp3 = []; tmp4 = [];
                tmp5 = []; tmp6 = [];
                tmp7 = []; tmp8 = []; tmp9 = [];
                % Concatenate features in each window to create a 
                % 1D feature vector for each block
                for k = 1:stride:blockDim-sampleSize+1
                    c = b(k:k+sampleSize-1);
                    tmp1 = [tmp1 mean(c)]; % Mean of the window
                    tmp4 = [tmp4 max(c)-min(c)]; % Peak-to-peak
                    tmp5 = [tmp5 skewness(c)]; % Skew
                    tmp6 = [tmp6 kurtosis(c)]; % Kurtosis
                    tmp7 = [tmp7 var(c)]; % Activity
                    tmp8 = [tmp8 std(diff(c))./std(c)]; % Mobility
                    tmp9 = [tmp9 std(diff(diff(c)))./std(diff(c))./(std(diff(c))./std(c))]; %Complexity
                end
                tmp2 = [tmp2 d];
                tmp3 = [tmp3 log10(e)']; % PSD features
                % Adding channel response labels
                if y(i)
                    tmp1 = [1 tmp1];
                    tmp2 = [1 tmp2];
                    tmp3 = [1 tmp3];
                    tmp4 = [1 tmp4];
                    tmp5 = [1 tmp5];
                    tmp6 = [1 tmp6];
                    tmp7 = [1 tmp7];
                    tmp8 = [1 tmp8];
                    tmp9 = [1 tmp9];
                else
                    tmp1 = [0 tmp1];
                    tmp2 = [0 tmp2];
                    tmp3 = [0 tmp3];
                    tmp4 = [0 tmp4];
                    tmp5 = [0 tmp5];
                    tmp6 = [0 tmp6];
                    tmp7 = [0 tmp7];
                    tmp8 = [0 tmp8];
                    tmp9 = [0 tmp9];
                end
                B = [B;tmp1];
                C = [C;tmp2];
                D = [D;tmp3];
                E = [E;tmp4];
                F = [F;tmp5];
                G = [G;tmp6];
                H = [H;tmp7];
                I = [I;tmp8];
                J = [J;tmp9];
            end
        end
    end
end