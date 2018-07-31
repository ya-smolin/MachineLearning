%% Matlab options
clear
close all;
format shortG
%% data load and preprocess
[X, yV, headers] = loadDatasetLog();
[X, headers] = preProcessFeatures(X, headers);
[m, n] = size(X);
plotDataLog(X, yV, headers);
%% build model
[J, dJ, h] = logReg(yV, X);
[J, dJ] = regularization(J, dJ, m, n);
thetaV = gradientDescent(dJ, J, X);

%% validate model
%r2 = statisticsReg(yV, X, @h);
trainingSetAccuracy=statisticsLog(yV, X, thetaV, h);
plotBoundary2D(X, thetaV);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% logistic regression
%% TODO!!
function [J, dJ, h] = logReg(yV, X)
    [m, ~] = size(X);
    h=@(thetaV,X)1./(1+exp(-linearB(thetaV, X)));
    J=@(thetaV)-1/m*sum((1 - yV).*(1-h(thetaV,X)) + yV.*h(thetaV,X));
    dJ=@(Theta)(1/m)*X'*(h(Theta,X)-yV);
end

%% linear regression model
function [J, dJ, h] = linearReg(yV, X)
    [m, ~] = size(X);
    h=@linearB;
    J=@(thetaV)1/(2.*m).*sum((h(thetaV,X)-yV).^2);
    dJ=@(Theta)(1/m)*X'*(h(Theta,X)-yV);
end
function yV = linearB(thetaV, X)
    yV=X*thetaV;
end

%% ML tools
function Theta = gradientDescent(dJ, J, X)
    [~, n] = size(X);
    lr = 0.0003; %have to be with minus for maximization
    iter = 100000;
    thetaV0 = ones(n,1);
    Theta_next = thetaV0;
    learningStep = zeros(iter,1);
    %while norm(Theta_next - Theta_prev) > eps
    for i=1:iter
        Theta_prev = Theta_next;
        Theta_next = Theta_prev - lr*dJ(Theta_prev);
        learningStep(i) = J(Theta_prev);
    end
    Theta = Theta_next;
    subplot(1,2,2);
    plot(1:iter, learningStep, 'b-');
    title(['Cost function:' num2str(learningStep(end))]);
end

function [Jr, dJr] = regularization(J, dJ, m, n)
    lambda = 0.01;
    I = eye(n,n);
    I(1,1)=0;
    Jr=@(thetaV)J(thetaV) + 1/(2*m)*lambda*thetaV'*I*thetaV;
    dJr=@(thetaV)dJ(thetaV)+lambda*[0;thetaV(2:end)];
end
function r2 = statisticsReg(yEx, X, h)
    yPr = h(thetaV, X);
    resModel = sum((yEx - yPr).^2);
    resTotal = (length(yEx)-1)*var(yEx);
    r2 = 1 - resModel/resTotal;
end
function trainingSetAccuracy = statisticsLog(yV, X, thetaV, h)
   yP = round(h(thetaV, X));
   trainingSetAccuracy = mean(yV==yP);
end
%% feature functions
function [X, headers] = preProcessFeatures(X, headers)
    X = normalize(X);
    %X = addExtraFeatures(X);
    m=size(X,1);
    X = [ones(m,1) X];
    headers = ['bias', headers];
end
function X = addExtraFeatures(X)
    X = [X X.^2 X.^3 X.^4 X.^5];
end
function X = normalize(X)
    X = (X-mean(X))./std(X);
end
%% DataSet tools
%X[m x n]
%m - feature size, n - dataset size
function [X, yV, headers] = loadHwyData()
    dataset = load('accidents.mat', 'hwydata', 'hwyheaders');
    xFeatureNum=14;
    yFeatureNum=4;
    m = size(dataset.hwydata, 1);
    X = dataset.hwydata(:,xFeatureNum); 
    yV = dataset.hwydata(:,yFeatureNum);
    headers = {dataset.hwyheaders{xFeatureNum}, dataset.hwyheaders{yFeatureNum}};
end
function [X, yV, headers] = loadDatasetLog()
    dataset = load('ex2data1.txt');
    X = dataset(:,1:2); 
    yV = dataset(:,3);
    headers = {'x1', 'x2'};
end
%% plot tools
function plotDataLog(X, yV, headers)
    subplot(1,2,1);
    hold on
    grid on;
    %[~, n] = size(X);
    xlabel(headers{2});
    ylabel(headers{3});
    yCol = find(yV == 1);
    plot(X(yCol,2), X(yCol,3), 'b+');
    yCol = find(yV == 0);
    plot(X(yCol,2), X(yCol,3), 'ro');
end

function plotDataReg(X, yV, headers)
    %title('smt')
    hold on
    grid on;
    [~, n] = size(X);
    if n == 2
        xlabel(headers{2})
        ylabel(headers{3})
        plot(X(:,2), yV, 'bo');
    elseif n == 3
        xlabel(headers{2})
        ylabel(headers{3})
        zlabel(headers{4})
        plot3(X(:,2),X(:,3),yV, 'bo');
    else
        warning('cannot plot in more than 3D');
    end
end
%plot3DMesh(boundary, -10:0.1:10);
function plot3DMesh(fun, range)
    [Xp, Yp] = meshgrid(range);
    [m,n] = size(Xp);
    Zp = zeros(m,n);
    for i = 1:m
        for j = 1:n
             tmp = fun([Xp(i,j); Yp(i,j)]);
             Zp(i,j) = tmp(1);
        end
    end
    mesh(Xp,Yp,Zp);
end
function plotBoundary2D(X, thetaV)
    subplot(1,2,1);
    %boundaryPolynomial=@(x)polyval(flipud(thetaV),x);
    [m, n]=size(X);
    boundaryLine=@(x)-thetaV(1)/thetaV(3)-thetaV(2)/thetaV(3)*x;
    fplot(boundaryLine, [min(min(X(:,2))), max(max(X(:,2)))]);
    %ylim([min(yV) max(yV)]);
end