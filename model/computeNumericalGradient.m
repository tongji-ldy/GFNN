function numgrad = computeNumericalGradient(J, theta1, theta2)
%COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
%and gives us a numerical estimate of the gradient.
%   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
%   gradient of the function J around theta. Calling y = J(theta) should
%   return the function value at theta.

% Notes: The following code implements numerical gradient checking, and 
%        returns the numerical gradient.It sets numgrad(i) to (a numerical 
%        approximation of) the partial derivative of J with respect to the 
%        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
%        be the (approximately) the partial derivative of J with respect 
%        to theta(i).)
%    

perturb1 = zeros(size(theta1));
perturb2 = zeros(size(theta2));
theta = [theta1(:) ; theta2(:)];
numgrad = zeros(size(theta));
e = 1e-4;
[m, n] = size(theta1);
max = m*n;
for p = 1:numel(theta)
    % Set perturbation vector

    if p <= max
        perturb1(p) = e;
    else
        perturb2(p-max) = e;
    end
    
    loss1 = J(theta1 - perturb1, theta2 - perturb2);
    loss2 = J(theta1 + perturb1, theta2 + perturb2);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    if p <= max
        perturb1(p) = 0;
    else
        perturb2(p-max) = 0;
    end
end

end
