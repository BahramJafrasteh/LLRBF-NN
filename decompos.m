function [cent,sigma]=decompos(XX,inp,Hn)

temp=XX(:,1:(inp)*Hn);
cent=reshape(temp,Hn,inp);
sigma=XX(:,(inp)*Hn+1:(inp+1)*Hn);
