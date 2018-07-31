%% Initialization
clear ; close all; clc
fprintf('Loading movie ratings dataset.\n\n');

%  Load data
load ('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  From the matrix, we can compute statistics like average rating.
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', ...
    mean(Y(1, R(1, :))));

%  We can "visualize" the ratings matrix by plotting it with imagesc
% imagesc(Y);
% ylabel('Movies');
% xlabel('Users');

%% ============ Part 2: Collaborative Filtering Cost Function ===========
%  You will now implement the cost function for collaborative filtering.
%  To help you debug your cost function, we have included set of weights
%  that we trained on that. Specifically, you should complete the code in
%  cofiCostFunc.m to return J.

%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
load ('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%
params=[X(:) ; Theta(:)];
% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
    num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

[i,j]=find(R~=0);
J=1/2*sum((sum(Theta(j,:).*X(i,:),2)-Y(R~=0)).^2)...
    +lambda/2*sum(sum(Theta.^2))+lambda/2*sum(sum(X.^2));
%
% for i=1:size(X,1)
%     for k=1:size(X,2)
%         X_grad(i,k)= 1/2*sum((Theta(j,:)*X(i,:)'-Y(R~=0)).*Theta(j,k));
%     end;
% end;
%
% [i,j]=find(R~=0);
% for j=1:size(Theta,1)
%     for k=1:size(Theta,2)
%         X_grad(j,k)= 1/2*sum((X(i,:)*Theta(j,:)'-Y(R~=0)).*X(i,k));
%     end;
% end;
for i = 1:size(X,1)
    [~,us]=find(R(i,:)==1);
    for k= 1:size(X,2);
        X_grad(i, k) = sum((Theta(us, :)*X(i, :)'-Y(i, us)').*Theta(us, k));
    end
end

for j = 1:size(Theta,1)
    [kin,~]=find(R(:,j)==1);
    for k= 1:size(Theta,2);
        Theta_grad(j, k) = sum((X(kin, :)*Theta(j, :)'-Y(kin, j)).*X(kin, k));
    end
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

