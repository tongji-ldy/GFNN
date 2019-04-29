function [ Theta1, Theta2, J ] = gradientDescent( costFunction, ...
                                                  Theta1, Theta2, hidden_layer_size, options, ...
                                                  X, y)
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
lamda_rule = 0.6;
delta_lamda = 0.0001;
flag = false;

for iter = 1:num_iters
    [J, Theta1_grad, Theta2_grad, z2] = costFunction(Theta1, Theta2, hidden_layer_size);
    
    fprintf('Iteration: %4i | Cost: %4.6e\r', iter, J);
    
    if J <= error
        break; 
    end
    
    
    if flag
        %删除神经元（每迭代500次进行一次删除操作）
        if mod(iter, 500)==0
            delete_count = zeros(1,hidden_layer_size);
            for i=1:hidden_layer_size-1
                for j=i+1:hidden_layer_size
                    if sum(z2(:,j))/size(z2,1)>lamda_rule && abs(Theta2(i)-Theta2(j))<0.001% && sum(abs(z2(:,i)-z2(:,j)))/size(z2,1)<0.001
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
        if iter>100
            if sum(max(z2,[],2))/size(z2,1) > lamda_rule
                if lamda_rule < 0.9
                    lamda_rule = lamda_rule + delta_lamda;
                end
            else
                %增加神经元
                if hidden_layer_size < 100
                    Theta1_add = [mean(X(:,1)), 0.8, mean(X(:,2)), 0.8];
                    Theta2_add = mean(y);
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

