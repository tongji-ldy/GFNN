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
alpha = 0.01;
error = 1e-6;
lamda_rule = 0.7;
% theta_rule = 0.4;
delta_lamda = 0.01;
delta_theta = 0.01;
flag = true;
m = size(X,1);

for iter = 1:num_iters
    for k = 1:m
        X_one = X(k,:);
        y_one = y(k,:);
        [J, Theta1_grad, Theta2_grad, z2] = costFunction(Theta1, Theta2, hidden_layer_size, theta_rule);
        
        fprintf('Iteration: %4i | Cost: %4.6e\r', iter, J);
        
        if J <= error
            break;
        end
        
        if flag
            %删除神经元（每迭代500次进行一次删除操作）
            if iter > 100 && mod(iter, 100)==0
                delete_count = zeros(1,hidden_layer_size);
                for i=1:hidden_layer_size-1
                    for j=i+1:hidden_layer_size
                        temp = abs(exp(-(Theta1(k,j*2-1)-Theta1(k,i*2-1)).^2./sigma(k,i*2).^2));
                        if temp>lamda_rule && abs(Theta2(i)-Theta2(j))<0.001
                            delete_count(j)=1;
                        end
                    end
                end
                delete_index = find(delete_count==1);
                Theta1(delete_index,:)=[];
                Theta2(:,delete_index)=[];
                Theta1_grad(delete_index,:)=[];
                Theta2_grad(:,delete_index)=[];
                z2(:,delete_index)=[];
                hidden_layer_size = hidden_layer_size-length(delete_index);
            end
            
            %增加神经元（每迭代100次进行一次增加判断）
            if iter > 100 && mod(iter, 50) == 0
                if sum(max(z2,[],2))/size(z2,1) > lamda_rule
                    if lamda_rule < 0.9
                        lamda_rule = lamda_rule + delta_lamda;
                        theta_rule = theta_rule + delta_theta;
                    end
                end
                if sum(max(z2,[],2))/size(z2,1) < theta_rule
                    %增加神经元
                    if hidden_layer_size < 100
                        Theta1_add = [mean(X_one(:,1)), 0.8, mean(X_one(:,2)), 0.8];
                        Theta2_add = mean(y_one);
                        Theta1 = [Theta1 ; Theta1_add];
                        Theta2 = [Theta2 , Theta2_add];
                        Theta1_grad = [Theta1_grad ; zeros(size(Theta1_add))];
                        Theta2_grad = [Theta2_grad , zeros(size(Theta2_add))];
                        hidden_layer_size = hidden_layer_size+1;
                    end
                end
            end
            
            fprintf('hidden_layer: %4i | lamda: %4.6e\r', hidden_layer_size, lamda_rule);
        end
        
        Theta1 = Theta1 - alpha * Theta1_grad;
        Theta2 = Theta2 - alpha * Theta2_grad;
    end
end

end

