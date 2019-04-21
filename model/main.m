%% Machine Learning - Neural Network
%  BP网络拟合函数matlab实现

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 2;    % 输入层神经元数量
hidden_layer_size = 50;   % 隐含层神经元数量
output_layer_size = 1;    % 输出层神经元数量
trainRatio = 0.7;         % 训练集比例
valRatio = 0.15;          % 验证集比例
testRatio = 0.15;         % 测试集比例

%% =========== Part 1: Loading Training Data =============

% Load Training Data
% 数据每一行为一组输入，共m组输入数据
fprintf('Loading Training Data ...\n')
load('data.mat');
m = size(X, 1);

%归一化数据
[Xn, Xs] = mapminmax(X',0,1);
[yn, ys] = mapminmax(y',0,1);
Xn = Xn';
yn = yn';

%随机取训练集和测试集
[trainInd, valInd, testInd] =dividerand(m,trainRatio,valRatio,testRatio);

Xn_train = Xn(trainInd,:);
yn_train = yn(trainInd,:);
Xn_test = Xn(testInd,:);
yn_test = yn(testInd,:);

%% ================ Part 2: Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =============== Part 3: Check Gradients ===============

fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients(1,Xn,yn);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =================== Part 4: Training NN ===================

fprintf('\nTraining Neural Network... \n')

%  传播迭代次数
options = optimset('MaxIter', 10000);

%  正则化参数
lambda =0;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, Xn_train, yn_train, lambda);

%optimization method1:fmincg
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%optimization method2:gradientDesent
%[nn_params, cost] = gradientDescent(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 5: Implement Predict =================

pred = predict(Theta1, Theta2, Xn_test);

pred = mapminmax('reverse',pred',ys);

%% ================= Part 6: result analysis =================

%预期与实际结构对比图
figure(1)

% plot(X(testInd,:), pred', 'r.')
% plot3(X(testInd,1), X(testInd,2), pred', 'r.')
plot(pred', ':og')

hold on

% plot(X(testInd,:), y(testInd,:),'.');
% plot3(X(testInd,1), X(testInd,2), y(testInd,:),'.');
plot(y(testInd,:),'-*');

legend('预测输出','期望输出')

title('BP网络预测输出','fontsize',12)

ylabel('函数输出','fontsize',12)

xlabel('样本','fontsize',12)

%预测误差
error=pred'-y(testInd,:);

figure(2)

plot(error,'-*')

title('BP网络预测误差','fontsize',12)

ylabel('误差','fontsize',12)

xlabel('样本','fontsize',12)

figure(3)

plot((y(testInd,:)-pred')./pred','-*');

title('神经网络预测误差百分比')

%误差总和
fprintf('\nTraining sum Accuracy: %f\n', sum(abs(error)));
