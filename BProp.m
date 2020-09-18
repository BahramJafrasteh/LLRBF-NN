%fitness function for wnnpso and backpropagation

function [fitbest,GlobalParams]=BProp(GlobalParams,Xtr,Ytr,Xv,Yv,inp,Hn,MaxIter)

eta=0.001;
% far=1.75;%1.75
if nargin<8
    MaxIter=2000;
end
% [vij,aj,bj,wjk]=decompose(pop,Hn,inp);
objfunc='fitrbf1';
fitnessvalid(1)=feval(objfunc,GlobalParams,Xv,Yv,Hn);
% bias=rand(1,Hn+1);
[N,~] = size (Xtr);
%add bias


%Backpropagation algorithm

roundci=zeros(Hn,inp);
% MaxIter=5000;
fitold=fitrbf1(GlobalParams,Xtr,Ytr,Hn);
tG=GlobalParams;
fitbest=fitold;M=1;
for iter=1:MaxIter
    %     eta=0.001;
    [teta,cent,sigma]=decompos0(tG,inp,Hn);
    W=(reshape(teta,Hn,inp+1));
    [~,~,Er,phi,~,~,distc]=fitrbf1(GlobalParams,Xtr,Ytr,Hn);
    Er=Er/N;
    
    for j=1:inp
        
        for i=1:Hn
            if j==1
                roundwi(i,1)=-Er'*phi(:,i);
            end
            roundwi(i,j+1)= -(Er.*Xtr(:,j))'*phi(:,i);
        end
        
    end
    
    tW=W-eta*roundwi;
    % tW=W;
    % tcent=cent-eta*roundci;
    tcent=cent;
    % tsig=sigma-eta*roundsig;
    tsig=sigma;
    tG=[reshape(tW,(inp+1)*Hn,1)',reshape(tcent,1,inp*Hn),tsig];
    fit=fitrbf1(tG,Xtr,Ytr,Hn);
    if fit<fitbest
        PG=GlobalParams;
        PF=fitbest;
        GlobalParams=tG;
        fitbest=fit;
        %                 eta=eta/0.1;
    else
        %         fitbest=fitold;
        tG=GlobalParams;
        
        eta=eta*0.1;
        %                 eta=eta*1.01;
    end
    if eta<1e-4
        eta=0.001;
    end
    %     df = Y.*(1-Y);%differential
    
    
    if (fitold-fit<0.001 || fit>1e2) && iter~=1
        %             fit=fitold;
        eta=eta*5;
        %         break
    end
    fitnessvalid(iter+1)=feval(objfunc,GlobalParams,Xv,Yv,Hn);
    
    if fitnessvalid(iter+1)-fitnessvalid(iter)>0.1
        %         fitbest=PF;
        %         GlobalParams=PG;
        break
    end
    
    %     end
    fitold=fit;
    if iter>M
        fprintf('Number of Iter=%d FitB=%d \n' , iter, fitbest)
        M=M+500;
    end
    
end

