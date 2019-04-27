function [J, Theta1_grad, Theta2_grad, z2] = nnCostFunction(Theta1, Theta2, ...
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

% Setup some useful variables
m = size(X, 1);

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 

%当输入为X时，计算神经网络中每一层的输出值（输出层的输出为a3）。
a1 = X;
x_aver = Theta1(:,1:2:end);
sigma = Theta1(:,2:2:end);

z2 = zeros(m,hidden_layer_size);
theta_rule = 0;
a2 = zeros(m,hidden_layer_size);
for i=1:m
    for j=1:hidden_layer_size
        z2(i,j) = prod(exp(-(a1(i,:)-x_aver(j,:)).^2./sigma(j,:).^2),2);
        if z2(i,j)-theta_rule>=0
            a2(i,j) = z2(i,j);
        end
    end
end

a3 = zeros(m,1);
for i=1:m
    [M, index] = max(a2(i,:));
    if M > 0
        a3(i) = a2(i,:) * Theta2'./sum(a2(i,:),2);
    else
        a3(i) = Theta2(index);
    end
end

%计算CostFunction的输出值J：
J = (a3 - y)'*(a3 - y)/(2*m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
temp1 = zeros(size(Theta2));
temp2 = zeros(size(Theta1));
for k=1:m
    for l=1:hidden_layer_size
        temp1(l) = ((a3(k)-y(k))./sum(a2(k,:),2)).*a2(k,l);
        for i=1:input_layer_size
            temp2(l,i*2-1) = temp1(l).*(Theta2(l)-a3(k)).*2*(X(k,i)-x_aver(l,i))./sigma(l,i).^2;
            temp2(l,i*2) = temp1(l).*(Theta2(l)-a3(k)).*2*(X(k,i)-x_aver(l,i)).^2./sigma(l,i).^3;
        end
    end
    Delta1 = Delta1 + temp2;
    Delta2 = Delta2 + temp1;
end

Theta1_grad = 1/m * Delta1;
Theta2_grad = 1/m * Delta2;

% Part 3: Implement regularization with the cost function and gradients.

temp1 = Theta1.*Theta1;
temp2 = Theta2.*Theta2;
J = J + lambda/(2*m)*(sum(temp1(:)) + sum(temp2(:)));
Theta1_grad = Theta1_grad + lambda/m .* Theta1;
Theta2_grad = Theta2_grad + lambda/m .* Theta2;

end
