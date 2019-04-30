% X = sort(rand(2000,1)*100);
% y = sin(X./(2*pi))+1;

X = rand(2000,2)*10;
y = 2*X(:,1)-sin(X(:,2));

%plot3(X(:,1),X(:,2),y,'.');

save('E:\matlab\GFNN\model\data.mat','X','y');