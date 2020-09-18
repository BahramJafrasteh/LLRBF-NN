function [W,cent,sig]=decompos0(XX,inp,Hn)
% Weights=XX(:,1:(inp+Ono)*Hn+(Hn^2)*(HiddenLayers-1));
W=XX(:,1:(inp+1)*Hn);
aijt=XX(:,(inp+1)*Hn+1:(2*inp+1)*Hn);
sig=abs(XX(:,(2*inp+1)*Hn+1:2*(inp+1)*Hn));
% landa=XX(:,2*(inp+1)*Hn+1);
% V=XX(:,(3*inp+1)*Hn+1:(3*inp+1)*Hn+inp);
% bl=XX(:,(3*inp+1)*Hn+inp+1);
cent=(reshape(aijt,Hn,inp));
% sig=(reshape(bijt,Hn,inp))';
% for i=1:inp
%     dilat(i,:)=aijt(:,(i-1)*Hn+1:i*Hn);
%     trans(i,:)=bijt(:,(i-1)*Hn+1:i*Hn);
% end
% W=(reshape(Wt,Hn,inp+1))';
% for i=1:inp+1
%     W(i,:)=Wt(:,(i-1)*Hn+1:i*Hn);
% end

end