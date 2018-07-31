function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    N=length(theta);
    theta_old=theta;
    for j=1:N
        theta(j)=theta(j)-alpha*sum(1/m*(h(X,theta_old)-y).*X(:,j));
    end
    
    
    
  
    % ============================================================
    
    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);
    
end

end

function h=h(X,theta)
h=0;
N=size(X,2);
for j=1:N
    h=h+X(:,j).*theta(j);
end
end