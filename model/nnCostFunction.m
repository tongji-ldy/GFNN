function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 

%当输入为X时，计算神经网络中每一层的输出值（输出层的输出为a3）。
X = [ones(m,1) X];%加上偏置项bias unit
a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2';
a3 = z3;

%计算CostFunction的输出值J：
J = (a3 - y)'*(a3 - y)/(2*m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

delta3 = a3 - y;
temp = delta3 * Theta2;
delta2 = temp(:,2:end) .* sigmoidGradient(z2);
Delta1 = Delta1 + delta2' * X;
Delta2 = Delta2 + delta3' * a2;

Theta1_grad = 1/m * Delta1;
Theta2_grad = 1/m * Delta2;

% Part 3: Implement regularization with the cost function and gradients.

temp1 = Theta1(:,2:end).*Theta1(:,2:end);
temp2 = Theta2(:,2:end).*Theta2(:,2:end);
J = J + lambda/(2*m)*(sum(temp1(:)) + sum(temp2(:)));
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m .* Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m .* Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
