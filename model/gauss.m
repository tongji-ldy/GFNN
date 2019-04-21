function g = gauss( x, x_aver, sigma )
%GAUSSIAN 高斯函数
%   高斯函数的a默认为1

g = exp(-(x-x_aver).^2./sigma.^2);

end

