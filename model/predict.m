function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
hidden_layer_size = size(Theta1, 1);   % 隐含层神经元数量
output_layer_size = size(Theta2, 1);    % 输出层神经元数量

% You need to return the following variables correctly 
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

p = a3;

% =========================================================================

end
