
                     %%%%% Randomly select i examples from the training set and  %%%%% 
                     %%%%% i examples from the cross validation set. And then    %%%%%
                     %%%%% learn the parameters θ using the randomly chosen      %%%%%
                     %%%%% training set and evaluate the parameters θ on the     %%%%%
                     %%%%% randomly chosen training set and cross validation set.%%%%%


%% Initialization
clear ; close all; clc

% Loading data
fprintf("Loading data from ex5data1.mat...\n")
load("ex5data1.mat");

%% ===============  We should combine taring and testing set as original training set has only    =================
%% ===============  12 examples which is half of that of valiadtion set. With so few training     =================
%% ===============  examples and 9 degree problem we don't have enough training examples and      =================
%% ===============  our results are not going to be very consistent from statistical standpoint   =================

% Combining training and testing set
X = [X; Xtest;];
y = [y; ytest;];

% Adding polynomial features
X = polyFeatures(X,9);
Xval = polyFeatures(Xval,9);

% Normalizing Features
X = featureNormalize(X);
Xval = featureNormalize(Xval);

% Size of X
m = size(X,1);

% Size of Xval
n = size(Xval,1);

% lambda
lambda = 0.07;


%% ------------------------------------------------------------------------------------------------------------- %%



%% ===============  Loop to calculate error on training and validation sets of different sizes    =================
%% ===============  For each size a random sample is taken and its error is calculated. The       =================
%% ===============  error for each size is then the average of all the errors for random samples  =================

error_X = zeros(12,1);
error_Xval = zeros(12,1);

lenRand = 50;


for i = 1:12
  
    error_iter_X = zeros(lenRand,1);
    error_iter_Xval = zeros(lenRand,1);
   
    % Calculates error for random samples of X and Xval 50 times
    for j = 1:lenRand
     
        rand_indices = randperm(n);
    
        X_rand = X(rand_indices(1:i),:);
        y_rand = y(rand_indices(1:i),:);

        Xval_rand = Xval(rand_indices(1:i),:);
        yval_rand = yval(rand_indices(1:i),:);

        [theta] = trainLinearReg([ones(i,1) X_rand], y_rand, lambda);

        error_iter_X(j) = linearRegCostFunction([ones(i,1) X_rand], y_rand(1:i,:), theta, lambda);
        error_iter_Xval(j) = linearRegCostFunction([ones(i,1) Xval_rand], yval_rand(1:i,:), theta, lambda);
         
        
    end
    
    % Average error for random samples of size i in 50 iterations
    error_X(i) = sum(error_iter_X) / 50;
    error_Xval(i) = sum(error_iter_Xval) / 50;

end

   
% Printing training and testing error

fprintf("# Training Example\tTraining Error\tCross_Validation Error\n");
for i = 1:12
    fprintf("  \t%d\t\t%f\t%f\n", i, error_X(i), error_Xval(i));
end




plot(1:12, error_X, 'LineWidth', 3, 1:12, error_Xval, 'LineWidth', 3);
title('Polynomial Regression Learning Curve (lambda = 0.07)')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])


  











                       
