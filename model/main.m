%% GFNN - REGRESSION
%  GFNN拟合函数matlab实现

%% Initialization
clear ; close all; clc

%% Setup the parameters
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
[trainInd, valInd, testInd] = dividerand(m,trainRatio,valRatio,testRatio);

Xn_train = Xn(trainInd,:);
yn_train = yn(trainInd,:);
Xn_test = Xn(testInd,:);
yn_test = yn(testInd,:);

%% ================ Part 2: Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size*2, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);

% initial_Theta1(:,1) = mean(Xn_train(:,1));
% initial_Theta1(:,3) = mean(Xn_train(:,2));
% initial_Theta1(:,2:2:end) = 0.8;%初始高斯函数宽度

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

%  正则化及匹配度参数
lambda =0;
theta_rule = 0.4;

% Create "short hand" for the cost function to be minimized
costFunction = @(p1, p2, p3, p4, p5, p6) nnCostFunction(p1, p2, input_layer_size, p3, ...
                                        output_layer_size, p4, p5, lambda, p6);

%optimization method1:fmincg
%[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%optimization method2:gradientDesent
[Theta1, Theta2, cost] = gradientDescent(costFunction, initial_Theta1, initial_Theta2, ...
                                         hidden_layer_size, options, Xn_train, yn_train, theta_rule);

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

title('GFNN预测输出','fontsize',12)

ylabel('函数输出','fontsize',12)

xlabel('样本','fontsize',12)

%预测误差
error=pred'-y(testInd,:);

figure(2)

plot(error,'-*')

title('GFNN预测误差','fontsize',12)

ylabel('误差','fontsize',12)

xlabel('样本','fontsize',12)

figure(3)

plot((y(testInd,:)-pred')./pred','-*');

title('神经网络预测误差百分比')

%误差总和
fprintf('\nTraining sum Accuracy: %f\n', sum(abs(error)));
