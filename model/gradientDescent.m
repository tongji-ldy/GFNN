function [ nn_params, J ] = gradientDescent( costFunction, nn_params, options )
%gradient Descent

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
    num_iters = options.MaxIter;
else
    num_iters = 100;
end

%学习步长及设置精度
alpha = 0.01;
error = 0.0001;

for iter = 1:num_iters
    [J, grad] = costFunction(nn_params);
    
    fprintf('Iteration: %4i | Cost: %4.6e\r', iter, J);
    
    if (J < error)
        break;
    end
    
    nn_params = nn_params - alpha * grad;
    
end

end

