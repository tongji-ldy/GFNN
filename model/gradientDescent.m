function [ Theta1, Theta2, J ] = gradientDescent( costFunction, ...
                                                            Theta1, Theta2, hidden_layer_size, options )
%gradient Descent

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
    num_iters = options.MaxIter;
else
    num_iters = 100;
end

%学习步长及设置精度
alpha = 0.01;
error = 0.001;
lamda_rule = 0.7;
delta_lamda = 0.01;
Theta1_add = randInitializeWeights(size(Theta1,2)-1, 1);
Theta2_add = randInitializeWeights(1-1, size(Theta2,1));

for iter = 1:num_iters
    [J, Theta1_grad, Theta2_grad, z2] = costFunction(Theta1, Theta2, hidden_layer_size);
    
    fprintf('Iteration: %4i | Cost: %4.6e\r', iter, J);
    
    if J <= error
        break;
    elseif max(z2) > lamda_rule
        lamda_rule = lamda_rule + delta_lamda;
    else
        %增加神经元
        Theta1 = [Theta1 ; Theta1_add];
        Theta2 = [Theta2 , Theta2_add];
        Theta1_grad = [Theta1_grad ; zeros(size(Theta1_add))];
        Theta2_grad = [Theta2_grad , zeros(size(Theta2_add))];
        hidden_layer_size = hidden_layer_size+1;
    end
    
    Theta1 = Theta1 - alpha * Theta1_grad;
    Theta2 = Theta2 - alpha * Theta2_grad;
    
end

end

