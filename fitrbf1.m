function [fit,Yh,E,sai,sait,teta,distc]=fitrbf1(pop,X,Y,Hn)
% load temp
[N,inp]=size(X);
for L=1:size(pop,1)
    [teta,cent,sigma]=decompos0(pop(L,:),inp,Hn);
    sigma=abs(sigma);
    distanc=zeros(N,Hn);
    
    p=1*ones(1,Hn);
    
    for j=1:Hn
        sums=0;
        sums2=0;
        landa=2;
        for k=1:inp
            sums=sums+((X(:,k)-cent(j,k)).^2);
            sums2=sums2+((X(:,k)-cent(j,k)));
        end
        distc(:,j)=sums;
        q=0.5;
        
        phio=exp(-sums.*((sigma(j)+0.5)^2));
        sai(:,j)=((sigma(j)^2))*phio.*(erfc(-sums2));
        
    end
    sai=real(sai);
    
    sait=[];
    
    k=1;
    for i=1:inp
        
        for j=1:Hn
            sait(:,k)=sai(:,j).*X(:,i);
            k=k+1;
        end
    end
    sait=[sai,sait];
    
    if nargin<5
        % teta=(Y)'/sait';
        Yh=sait*teta';
        
    elseif size(teta,1)>1
        Yh=sait*teta(:,L);
        %         Yh=(sait*teta(:,L))./sum(sait,2);
    else
        Yh=sait*teta';
        % Yh=(sait*teta')./sum(sait,2);
    end
    
    E = Y - Yh;%error
    
    tem= (mean(E.^2));
    if ~isnan(tem)
        fit(L)=tem;
    else
        fit(L)=inf;
    end
end