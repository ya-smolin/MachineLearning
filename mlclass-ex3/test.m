% Theta1=random('Normal',0,1,2,5);
% Theta2=random('Normal',0,1,3,6);
% thetaVec = [Theta1(:) ; Theta2(:)];
% Theta2Rec=reshape(thetaVec(11:28),3,6);
J=@(t)3*t.^4+4;
eps=0.01;
t=1;
ans=(J(t+eps)-J(t-eps))/(2*eps);
