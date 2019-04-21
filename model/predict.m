function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
% p = zeros(size(X, 1), 1);
% 
% h1 = sigmoid([ones(m, 1) X] * Theta1');
% h2 = [ones(m, 1) h1] * Theta2';
% p = h2;

input_layer_size  = 2;    % 输入层神经元数量
hidden_layer_size = 50;   % 隐含层神经元数量
output_layer_size = 1;    % 输出层神经元数量

X = [ones(m,1) X];%加上偏置项bias unit
a1 = X;
x_aver = Theta1(:,1:2:end);
sigma = Theta1(:,2:2:end);

z2 = zeros(m,hidden_layer_size);
for i=1:m
    for j=1:hidden_layer_size
        z2(i,j) = prod(exp(-(a1(i,:)-x_aver(j,:)).^2./sigma(j,:).^2),2);
    end
end

theta = 0;
% lamda = 0.7;
a2 = zeros(m,hidden_layer_size);
for i=1:m
    for j=1:hidden_layer_size
        if z2(i,j)-theta>=0
            a2(i,j) = z2(i,j);
        end
    end
end

a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2';
a3 = z3./sum(a2,2);

p=a3;

% =========================================================================

end
