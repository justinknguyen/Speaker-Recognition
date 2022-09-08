%define the number of Gaussian invariants - could be modified
No_of_Gaussians=5;
%Reading in the data 
%Use wavread from matlab 
disp('-------------------------------------------------------------------');
disp('                    Speaker recognition Demo');
disp('                    using GMM');
disp('-------------------------------------------------------------------');

%-----------reading in the training data----------------------------------
training_data1=audioread('voice memos/wav/Lukas - 40 sec.wav');
training_data2=audioread('voice memos/wav/Josh - 38 sec.wav');
training_data3=audioread('voice memos/wav/Justin - 40 sec.wav');
training_data4=audioread('voice memos/wav/Aliah - 38 sec.wav');

%------------reading in the test data-------------------------------------
[testing_data1,Fs]=audioread('voice memos/wav/Lukas - 13 sec.wav');
testing_data2=audioread('voice memos/wav/Josh - 13 sec.wav');
testing_data3=audioread('voice memos/wav/Justin - 12 sec.wav');
testing_data4=audioread('voice memos/wav/Aliah - 13 sec.wav');
%not in training set (probe samples)
testing_data_p1=audioread('voice memos/wav/Adam (not in set) - 11 sec.wav');
testing_data_p2=audioread('voice memos/wav/Dan (not in set) - 14 sec.wav');

disp('Completed reading taining and testing data');

%Fs=8000;   %uncoment if you cannot obtain the feature number from wavread above

%-------------feature extraction------------------------------------------
training_features1=melcepst(training_data1,Fs);
training_features2=melcepst(training_data2,Fs);
training_features3=melcepst(training_data3,Fs);
training_features4=melcepst(training_data4,Fs);

disp('Completed feature extraction for the training data');

testing_features1=melcepst(testing_data1,Fs);
testing_features2=melcepst(testing_data2,Fs);
testing_features3=melcepst(testing_data3,Fs);
testing_features4=melcepst(testing_data4,Fs);
%not in training set (probe samples)
testing_features_p1=melcepst(testing_data_p1,Fs);
testing_features_p2=melcepst(testing_data_p2,Fs);

disp('Completed feature extraction for the testing data');

%-------------training the input data using GMM-------------------------
%training input data, and creating the models required
disp('Training models with the input data');

[mu_train1,sigma_train1,c_train1]=gmm_estimate(training_features1',No_of_Gaussians);
disp('Completed Training Speaker 1 model');

[mu_train2,sigma_train2,c_train2]=gmm_estimate(training_features2',No_of_Gaussians);
disp('Completed Training Speaker 2 model');

[mu_train3,sigma_train3,c_train3]=gmm_estimate(training_features3',No_of_Gaussians);
disp('Completed Training Speaker 3 model');

[mu_train4,sigma_train4,c_train4]=gmm_estimate(training_features4',No_of_Gaussians);
disp('Completed Training Speaker 4 model');

disp('Completed Training ALL Models');

%-------------------------testing against the input data-------------- 
%testing against the first model
[lYM,lY]=lmultigauss(testing_features1', mu_train1,sigma_train1,c_train1);
A(1,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train1,sigma_train1,c_train1);
A(1,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train1,sigma_train1,c_train1);
A(1,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features4', mu_train1,sigma_train1,c_train1);
A(1,4)=mean(lY);

%testing against the second model
[lYM,lY]=lmultigauss(testing_features1', mu_train2,sigma_train2,c_train2);
A(2,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train2,sigma_train2,c_train2);
A(2,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train2,sigma_train2,c_train2);
A(2,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features4', mu_train2,sigma_train2,c_train2);
A(2,4)=mean(lY);

%testing against the third model
[lYM,lY]=lmultigauss(testing_features1', mu_train3,sigma_train3,c_train3);
A(3,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train3,sigma_train3,c_train3);
A(3,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train3,sigma_train3,c_train3);
A(3,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features4', mu_train3,sigma_train3,c_train3);
A(3,4)=mean(lY);

%testing against the fourth model
[lYM,lY]=lmultigauss(testing_features1', mu_train4,sigma_train4,c_train4);
A(4,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train4,sigma_train4,c_train4);
A(4,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train4,sigma_train4,c_train4);
A(4,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features4', mu_train4,sigma_train4,c_train4);
A(4,4)=mean(lY);

disp('Results in the form of confusion matrix for comparison');
disp('Each column i represents the test recording of Speaker i');
disp('Each row i represents the training recording of Speaker i');
disp('The diagonal elements corresponding to the same speaker');
disp('-------------------------------------------------------------------');
A
disp('-------------------------------------------------------------------');
threshold = -15.7; %threshold based on scores
%determining match error rate
match = 0;
truematch = 4; %expected number of true matches 
for x = 1:4 %cycle through matrix
    for y = 1:4
        if (A(x,y) > threshold)
            match = match + 1;
        end
    end
end
disp('Match Error Rate:');
errorrate = (truematch - match)/(match)*100;
fprintf('%.2f%%\r\n',abs(errorrate))

%determining non-match error rate
nonmatch = 0;
truenonmatch = 12; %expected number of true non-matches 
for x = 1:4 %cycle through matrix
    for y = 1:4
        if (A(x,y) < threshold)
            nonmatch = nonmatch + 1;
        end
    end
end
disp('Non-Match Error Rate:');
errorrate = (truenonmatch - nonmatch)/(nonmatch)*100;
fprintf('%.2f%%\r\n',abs(errorrate))

% confusion matrix in color
figure; imagesc(A); colorbar;
title("Similarity Score Confusion Matrix")
disp('-------------------------------------------------------------------');

%true/false match rate
fmatch = 0;
tmatch = 0;
falsematch = 12; %expected number of false matches 
truematch = 4; %expected number of true matches 
for x = 1:4 %cycle through matrix
    for y = 1:4
        if (A(x,y) > threshold)
            if (x==y)
                tmatch = tmatch + 1;
            end
            if (x~=y)
                fmatch = fmatch + 1;
            end
        end
    end
end
disp('False Match Rate:');
falserate = (fmatch)/(falsematch)*100;
fprintf('%.2f%%\r\n',abs(falserate))
disp('True Match Rate:');
truerate = (tmatch)/(truematch)*100;
fprintf('%.2f%%\r\n',abs(truerate))

%false non-match rate
fnmatch = 0;
falsenonmatch = 4; %expected number of false non-matches 
for x = 1:4 %cycle through matrix
    for y = 1:4
        if (A(x,y) < threshold)
            if (x==y)
                fnmatch = fnmatch + 1;
            end
        end
    end
end
disp('False Non-Match Rate:');
fnmrate = (fnmatch)/(falsenonmatch)*100;
fprintf('%.2f%%\r\n',abs(fnmrate))

%-------------------------testing first probe sample against the input data--------------
%testing against the first model
[lYM,lY]=lmultigauss(testing_features1', mu_train1,sigma_train1,c_train1);
A1(1,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p1', mu_train1,sigma_train1,c_train1);
A1(1,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p1', mu_train1,sigma_train1,c_train1);
A1(1,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p1', mu_train1,sigma_train1,c_train1);
A1(1,4)=mean(lY);

%testing against the second model
[lYM,lY]=lmultigauss(testing_features_p1', mu_train2,sigma_train2,c_train2);
A1(2,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train2,sigma_train2,c_train2);
A1(2,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p1', mu_train2,sigma_train2,c_train2);
A1(2,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p1', mu_train2,sigma_train2,c_train2);
A1(2,4)=mean(lY);

%testing against the third model
[lYM,lY]=lmultigauss(testing_features_p1', mu_train3,sigma_train3,c_train3);
A1(3,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p1', mu_train3,sigma_train3,c_train3);
A1(3,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train3,sigma_train3,c_train3);
A1(3,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p1', mu_train3,sigma_train3,c_train3);
A1(3,4)=mean(lY);

%testing against the fourth model
[lYM,lY]=lmultigauss(testing_features_p1', mu_train4,sigma_train4,c_train4);
A1(4,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p1', mu_train4,sigma_train4,c_train4);
A1(4,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p1', mu_train4,sigma_train4,c_train4);
A1(4,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features4', mu_train4,sigma_train4,c_train4);
A1(4,4)=mean(lY);

disp('Results in the form of confusion matrix for comparison');
disp('Each column i represents the test recording of Speaker i');
disp('Each row i represents the training recording of Speaker i');
disp('The diagonal elements corresponding to the same speaker');
disp('-------------------------------------------------------------------');
A1
disp('-------------------------------------------------------------------');
%determining match error rate
match = 0;
truematch = 4; %expected number of true matches 
for x = 1:4 %cycle through matrix
    for y = 1:4
        if (A1(x,y) > threshold)
            match = match + 1;
        end
    end
end
disp('Match Error Rate:');
errorrate = (truematch - match)/(match)*100;
fprintf('%.2f%%\r\n',abs(errorrate))

%determining non-match error rate
nonmatch = 0;
truenonmatch = 12; %expected number of true non-matches 
for x = 1:4 %cycle through matrix
    for y = 1:4
        if (A1(x,y) < threshold)
            nonmatch = nonmatch + 1;
        end
    end
end
disp('Non-Match Error Rate:');
errorrate = (truenonmatch - nonmatch)/(nonmatch)*100;
fprintf('%.2f%%\r\n',abs(errorrate))

% confusion matrix in color
figure; imagesc(A1); colorbar;
title("Similarity Score Confusion Matrix")
disp('-------------------------------------------------------------------');

%-------------------------testing second probe sample against the input data--------------
%testing against the first model
[lYM,lY]=lmultigauss(testing_features1', mu_train1,sigma_train1,c_train1);
A2(1,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p2', mu_train1,sigma_train1,c_train1);
A2(1,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p2', mu_train1,sigma_train1,c_train1);
A2(1,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p2', mu_train1,sigma_train1,c_train1);
A2(1,4)=mean(lY);

%testing against the second model
[lYM,lY]=lmultigauss(testing_features_p2', mu_train2,sigma_train2,c_train2);
A2(2,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train2,sigma_train2,c_train2);
A2(2,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p2', mu_train2,sigma_train2,c_train2);
A2(2,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p2', mu_train2,sigma_train2,c_train2);
A2(2,4)=mean(lY);

%testing against the third model
[lYM,lY]=lmultigauss(testing_features_p2', mu_train3,sigma_train3,c_train3);
A2(3,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p2', mu_train3,sigma_train3,c_train3);
A2(3,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train3,sigma_train3,c_train3);
A2(3,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p2', mu_train3,sigma_train3,c_train3);
A2(3,4)=mean(lY);

%testing against the fourth model
[lYM,lY]=lmultigauss(testing_features_p2', mu_train4,sigma_train4,c_train4);
A2(4,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p2', mu_train4,sigma_train4,c_train4);
A2(4,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features_p2', mu_train4,sigma_train4,c_train4);
A2(4,3)=mean(lY);
[lYM,lY]=lmultigauss(testing_features4', mu_train4,sigma_train4,c_train4);
A2(4,4)=mean(lY);

disp('Results in the form of confusion matrix for comparison');
disp('Each column i represents the test recording of Speaker i');
disp('Each row i represents the training recording of Speaker i');
disp('The diagonal elements corresponding to the same speaker');
disp('-------------------------------------------------------------------');
A2
disp('-------------------------------------------------------------------');
%determining match error rate
match = 0;
truematch = 4; %expected number of true matches 
for x = 1:4 %cycle through matrix
    for y = 1:4
        if (A2(x,y) > threshold)
            match = match + 1;
        end
    end
end
disp('Match Error Rate:');
errorrate = (truematch - match)/(match)*100;
fprintf('%.2f%%\r\n',abs(errorrate))

%determining non-match error rate
nonmatch = 0;
truenonmatch = 12; %expected number of true non-matches 
for x = 1:4 %cycle through matrix
    for y = 1:4
        if (A2(x,y) < threshold)
            nonmatch = nonmatch + 1;
        end
    end
end
disp('Non-Match Error Rate:');
errorrate = (truenonmatch - nonmatch)/(nonmatch)*100;
fprintf('%.2f%%\r\n',abs(errorrate))

% confusion matrix in color
figure; imagesc(A2); colorbar;
title("Similarity Score Confusion Matrix")
disp('-------------------------------------------------------------------');