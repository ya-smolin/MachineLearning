%% Matlab options
clear
close all;
format shortG
%% data load and preprocess
modelName = 'linear';
dataS = loadDataset(modelName);
dataS.X = preProcessFeatures(dataS.X);
plotData(modelName, dataS);
%% build model
modelS = model(modelName,dataS);
%modelS = regularization(modelS);
methodS = struct('name', 'gradient', 'iter', 1000, 'lr', 0.003, 'thetaV0', ones(modelS.n, 1));
options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
thetaV = fminunc(@(theta)fminFunc(theta, modelS),ones(modelS.n, 1),options);
%[thetaV, learningStep] = optimize(modelS, methodS);
%plotLearningStep(learningStep);
%% validate model
accuracy = statistics(modelName, dataS, modelS, thetaV);
plotBoundary2D(dataS.X, thetaV);

function [fun, grad] = fminFunc(theta, modelS)
    fun=modelS.J(theta);
    grad=modelS.dJ(theta);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% model
function modelS = model(name, dataS)
    yV=dataS.yV;
    X=dataS.X;
    [m,n]= size(X);
    modelS.m = m;
    modelS.n = n;
    linearB = @(thetaV, X)X*thetaV;
    if strcmp(name, 'logistic')
        h=@(thetaV,X)1./(1+exp(-linearB(thetaV, X)));
        modelS.J=@(thetaV)-1/m*sum((1 - yV).*log((1-h(thetaV,X))) + yV.*log(h(thetaV,X)));
        modelS.dJ=@(thetaV)(1/m)*X'*(h(thetaV,X)-yV);
        modelS.H=@(thetaV)(1/m)*X'*diag(h(thetaV, X).*(1-h(thetaV, X)))*X;
    elseif strcmp(name, 'linear')
        h=linearB;
        modelS.J=@(thetaV)1/(2.*m).*sum((h(thetaV,X)-yV).^2);
        modelS.dJ=@(Theta)(1/m)*X'*(h(Theta,X)-yV);
    elseif strcmp(name, 'poisson')
        h=@(thetaV, X)exp(linearB(thetaV, X));
        modelS.J=@(thetaV)sum(log(factorial(yV))-yV.*linearB(thetaV, X) + h(thetaV, X));
        modelS.dJ=@(Theta)X'*(h(Theta,X)-yV);
    modelS.h = h;
    modelS.name=name;
end
%% learning
function [thetaV, learningStep] = optimize(modelS, methodS)
isLearningCurve = false;
if(isfield(modelS, 'J'))
    isLearningCurve = true;
    learningStep = zeros(methodS.iter,1);
end
Theta_next = methodS.thetaV0;
for i=1:methodS.iter
    Theta_prev = Theta_next;
    if strcmp(methodS.name,'newton')
        Theta_next = Theta_prev - modelS.H(Theta_prev)\modelS.dJ(Theta_prev);
    elseif strcmp(methodS.name,'gradient')
        Theta_next = Theta_prev - methodS.lr*modelS.dJ(Theta_prev);
    end
    if isLearningCurve
        learningStep(i) = modelS.J(Theta_prev);
    end
end
thetaV = Theta_next;
end
%% regularization
function modelS = regularization(modelS)
    lambda = 0.1;
    I = eye(modelS.n,modelS.n);
    I(1,1)=0;
    modelS.J=@(thetaV)modelS.J(thetaV) + 1/(2*modelS.m)*lambda*thetaV'*I*thetaV;
    modelS.dJ=@(thetaV)modelS.dJ(thetaV)+lambda*[0;thetaV(2:end)];
    modelS.H=@(thetaV)modelS.H(thetaV)+lambda*[0;ones(length(thetaV)-1,1)];
end
%% evaluation
function accuracy = statistics(modelName, dataS, modelS, thetaV)
    yV = dataS.yV;
    X = dataS.X;
    h = modelS.h;
    if strcmp(modelName, 'linear')||strcmp(modelName, 'poisson')
        %%r2 evaluation
        yPr = h(thetaV, X);
        resModel = sum((yV - yPr).^2);
        resTotal = (length(yV)-1)*var(yV);
        accuracy = 1 - resModel/resTotal;
    elseif strcmp(modelName, 'logistic')
         yP = round(h(thetaV, X));
         accuracy = mean(yV==yP);
    end
end
%% feature functions
function X = preProcessFeatures(X)
    %X = normalize(X);
    %X = addExtraFeatures(X);
    m=size(X,1);
    X = [ones(m,1) X];
end
function X = addExtraFeatures(X)
    X = [X X.^2 X.^3];
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
function dataS = loadDataset(modelName)
    if strcmp(modelName, 'logistic')
        dataS.X = load('logistic_x.txt');
        dataS.yV = load('logistic_y.txt');
        dataS.yV(dataS.yV==-1)=0;    
    elseif strcmp(modelName, 'logistic1')
        %circle
        dataset = load('ex2data2.txt');
        dataS.X = dataset(:,1:2);
        dataS.yV = dataset(:,3);
    elseif strcmp(modelName, 'poisson')
        rng('default') % for reproducibility
        X = randn(100,7);
        mu = exp(X(:,[1 3 6])*[.4;.2;.3] + 1);
        y = poissrnd(mu);
        dataS.X = X(:,1);
        dataS.yV = y;
    elseif strcmp(modelName, 'linear') 
        dataset = load('ex1data1.txt');
        dataS.X = dataset(:,1);
        dataS.yV = dataset(:,2);
    end
end
%% plot tools
function plotData(modelName, dataS)
    %plot in feature space
    [m, n]=size(dataS.X);
    subplot(1,2,1);
    hold on
    grid on;
    if strcmp(modelName, 'logistic')
        %2D
        yCol = find(dataS.yV == 1);
        plot(dataS.X(yCol,2), dataS.X(yCol,3), 'b+');
        yCol = find(dataS.yV == 0);
        plot(dataS.X(yCol,2), dataS.X(yCol,3), 'ro');
    elseif strcmp(modelName, 'linear') || strcmp(modelName, 'poisson')
        if n == 2
            plot(dataS.X(:,2), dataS.yV, 'bo');
        elseif n == 3
            plot3(dataS.X(:,2),dataS.X(:,3),dataS.yV, 'bo');
        else
        	warning('cannot plot in more than 3D');
        end
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
    %boundary=@(x1,x2)thetaV(1)+ thetaV(2).*x1+thetaV(3).*x2+thetaV(4).*x1.^2+thetaV(5).*x2.^2+thetaV(6).*x1.^3+thetaV(7).*x2.^3;
    %boundarySym = sym(boundary);
    %ezplot(boundarySym,[min(min(X(:,2))) max(max(X(:,2))) min(X(:,3)) max(X(:,3))])
    
    %plot3DMesh(boundary,[min(min(X(:,2))) max(max(X(:,2))) min(X(:,3)) max(X(:,3))])
    %fcontour(boundary,[min(min(X(:,2))) max(max(X(:,2))) min(X(:,3)) max(X(:,3))]);
    %boundary=@(x)polyval(flipud(thetaV),x); %polynomial
    %boundary=@(x)-thetaV(1)/thetaV(3)-thetaV(2)/thetaV(3)*x; %line
    boundary = @(x1)exp(thetaV(1)+ thetaV(2).*x1);
    %boundarySym = sym(boundary);
    %ezplot(boundarySym,[min(min(X(:,2))) max(max(X(:,2)))]);
    fplot(boundary, [min(min(X(:,2))), max(max(X(:,2)))]);
    %ylim([min(X(:,3)) max(X(:,3))]);
end
function plotLearningStep(learningStep)
    subplot(1,2,2);
    iter=length(learningStep);
    plot(1:iter, learningStep, 'b-');
    title(['Cost function:' num2str(learningStep(end))]);
end