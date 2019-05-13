function [ Theta1, Theta2, Theta1_grad, Theta2_grad, hidden_layer_size, theta_rule, lamda_rule ] = organizeStructure( Theta1, Theta2, ...
                                                                Theta1_grad, Theta2_grad, hidden_layer_size, ...
                                                                X, y, z2, theta_rule, lamda_rule, iter, k )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

delta_lamda = 0.001;
delta_theta = 0.001;
flag = true;

if flag
    %删除神经元（每迭代500次进行一次删除操作）
    if iter > 100 && mod(iter, 100)==0
        delete_count = zeros(1,hidden_layer_size);
        for i=1:hidden_layer_size-1
            for j=i+1:hidden_layer_size
                temp = abs(prod(exp(-(Theta1(j,1:2:end)-Theta1(i,1:2:end)).^2./Theta1(i,2:2:end).^2),2));
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
        [M, index] = max(z2,[],2);
        if sum(M)/size(z2,1) > lamda_rule
            if lamda_rule < 0.9
                lamda_rule = lamda_rule + delta_lamda;
                theta_rule = theta_rule + delta_theta;
                Theta2(index) = y;
            end
        end
        if sum(M)/size(z2,1) < lamda_rule%theta_rule
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
end

end

