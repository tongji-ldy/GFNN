function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 

% You need to return the following variables correctly 
W = zeros(L_out, L_in);

% Initialize W randomly to break the symmetry while
% training the neural network.

epsilon_init = 10;%0.5;
W = rand(L_out, L_in) * 2 * epsilon_init - epsilon_init;

end
