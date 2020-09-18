function [X,Y,xcent,xhalf,ycent,yhalf] = prepdata(X,Y)
%Prepdata: Data pre-processing
% load temp
[N, ~] = size(X);

xmax = max(X)';
xmin = min(X)';

xcent = 0.5 * (xmax + xmin); % centter of X domain
xhalf = 0.5 * (xmax - xmin); % half larger of X domain

X = (X - (xcent * ones(1,N))') ./ (xhalf * ones(1,N))';

ymax = max(Y);
ymin = min(Y);

ycent = 0.5 * (ymax + ymin); % centter of Y domain
yhalf = 0.5 * (ymax - ymin); % half larger of Y domain

Y = (Y - ycent) ./ (yhalf * ones(N,1));
