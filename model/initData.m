% X = sort(rand(2000,1)*100);
% y = sin(X./(2*pi))+1;

X = rand(2000,2)*100;
%X = sort(rand(2000,2)*100);
y = 2*X(:,1).^2-X(:,2).^2;

%plot3(X(:,1),X(:,2),y,'.');

save('E:\matlab\ml_ex\machine-learning-ex4\ex4\data.mat','X','y');