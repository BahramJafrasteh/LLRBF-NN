function [X,Y] = postdata(X,Y,xcent,xhalf,ycent,yhalf)
%Prepdata: Data post-processing

[N, ~] = size(X);
X=(xhalf * ones(1,N))'.*X+(xcent * ones(1,N))';
Y=Y.*(yhalf * ones(N,1))+ycent;

