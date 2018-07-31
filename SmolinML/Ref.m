load iris_dataset.mat
irisInputs = irisInputs(:,1:100)';
yV = zeros(100,1);
yV(1:50,1) = 1;
yV(51:100,1) = 0;
x1d(1:50) = 0;
X = irisInputs(:,1);

X1 = X(find(yV == 1), 1);
plot(X1, x1d, 'b+');
hold on;
X0 = X(find(yV == 0), 1);
plot(X0, x1d, 'ro');

mu0 = mean(X0);
mu1 = mean(X1);


sigma0 = std(X0);
sigma1 = std(X1);
norm0 = @(x)normpdf(x, mu0, sigma0);
norm1 = @(x)normpdf(x, mu1, sigma1);
plot([mu0-sigma0, mu0+sigma0], repmat(norm0(mu0+sigma0),1, 2), 'r--');
plot([mu1-sigma1, mu1+sigma1], repmat(norm1(mu1+sigma1),1, 2), 'b--');
plot(repmat(mu0,1, 2), [norm0(mu0), 0], 'r--');
plot(repmat(mu1,1, 2), [norm1(mu1), 0], 'b--');

fplot(norm0, [min(X)*0.9, max(X)], 'r-');
fplot(norm1, [min(X)*0.9, max(X)], 'b-');

px=@(x)length(X0)./length(X)*norm0(x)+length(X1)./length(X)*norm1(x);
fplot(px, [min(X)*0.9, max(X)], 'y-');

phi = length(X1)./length(X);
py=@(y)phi.^y*(1-phi).^(1-y);

px_y=@(x)norm0(x).*norm1(x);
fplot(px_y, [min(X)*0.9, max(X)], 'g-');
py_x1=@(x)py(1).*px_y(x)./px(x);
fplot(py_x1, [min(X)*0.9, max(X)], 'g--');