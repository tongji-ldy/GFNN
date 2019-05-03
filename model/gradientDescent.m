function [ Theta1, Theta2, J ] = gradientDescent( costFunction, ...
                                                  Theta1, Theta2, hidden_layer_size, options, ...
                                                  X, y, theta_rule)
%gradient Descent

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
    num_iters = options.MaxIter;
else
    num_iters = 100;
end

%学习步长及设置精度
alpha = 0.003;
error = 1e-20;
m = size(X,1);
lamda_rule = 0.7;

for iter = 1:num_iters
    for k = 1:m
        X_one = X(k,:);
        y_one = y(k,:);
        
        [J, Theta1_grad, Theta2_grad, z2] = costFunction(Theta1, Theta2, hidden_layer_size, X_one, y_one, theta_rule);

        [ Theta1, Theta2, Theta1_grad, Theta2_grad, hidden_layer_size, theta_rule, lamda_rule ] = organizeStructure(Theta1, Theta2, Theta1_grad, Theta2_grad, hidden_layer_size, X_one, y_one, z2, theta_rule, lamda_rule, iter, k);
        
        Theta1 = Theta1 - alpha * Theta1_grad;
        Theta2 = Theta2 - alpha * Theta2_grad;
    end
    
     fprintf('Iteration: %4i | Cost: %4.6e\r', iter, J);
     fprintf('hidden_layer: %4i | lamda: %4.6e\r', hidden_layer_size, lamda_rule);
     
     if J <= error
         break;
     end
     
end

end

