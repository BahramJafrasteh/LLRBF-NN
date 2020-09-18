%%Local Linear wavelet neural network by pso algorithm
%Coding by Bahram Jafrasteh
%PhD student at IUT
%Date: 12/11/2014
%
%main
clc
clear all
runtime=30;      % The number of test time
GCs=[0,0,0,0];
for r=1:runtime
    save ('temPSO','r','runtime','GCs')
    clear
    load temPSO
    rng('default')
    rng('shuffle')
load esfordi_LLRBF
    X=[X,new];
    [X,~,xcent,xhalf,ycent,yhalf] = prepdata(X,Y);
    [N,inp]=size(X);
    Xtr=X(1:size(Xtr,1),:);
    Ytr=Y(1:size(Xtr,1),:);
    Xv=X(size(Xtr,1)+1:size(Xtr,1)+size(Xv,1),:);
    Yv=Y(size(Xtr,1)+1:size(Xtr,1)+size(Xv,1));
    Xt=X(size(Xtr,1)+size(Xv,1)+1:size(Xtr,1)+size(Xv,1)+size(Xt,1),:);
    Yt=Y(size(Xtr,1)+size(Xv,1)+1:size(Xtr,1)+size(Xv,1)+size(Xt,1));
    
    objfunc='fitrbf1';
    Maxiter=350;
    Maxc=20000;
    popsize=50;
    [~,inp]=size(Xtr);%number of membership function for each input
    Hn=15;%number of hidden nodes
    D=2*(inp+1)*Hn+1;
    wMax=0.9;
    wMin=0.1;
    change=1;
    MaxGl=4;%percent
    c1=2;c2=2;
    for k=1:popsize
            cent=rand(Hn,inp);

        centt=reshape(cent,1,inp*Hn);
        dist=max(max(pdist2(cent,cent)));
        %     sigma=(dist/sqrt(2*Hn))*ones(1,Hn);
        sigma=(1.2)*ones(1,Hn);
        W(:,k)=0.2*rand((inp+1)*Hn,1)-0.1;
        bias=rand;
        temp=[W(:,k)',centt,sigma,bias];
        pop(k,:)=[W(:,k)',centt,sigma,bias];
        %     [~,pop(k,:),W(:,k)] = trainrbf(Xtr,Ytr,pop(k,:),Hn,W(:,k),1);
        %     [~,~,~,hi]=fitrbf(pop(k,:),Xtr,Ytr,Hn,W);
        %     W=(Ytr'/hi')';
        
    end
    fit=feval(objfunc,pop,Xtr,Ytr,Hn);
    v=(0.2)*rand(popsize,D)-0.1;
    
    %%
    pbest=fit;
    [gbest, lab1]=min(fit);
    xgbest=pop(lab1,:);
    wbest=W(:,lab1);
    vmax=20;
    % c1=.2;
    % c2=.2;
    xpbest=pop;
    temptemp=[];
    temptemp1=[];
    % figure(1)
    gbest1=gbest;
    II=0;iter1=1;count=0;M=1;
    for iter=1:Maxiter
        
        %generalisation loss
        
        %         [~,~,~,~,~,wbest]=fitrbf(xgbest,Xtr,Ytr,Hn);
        fitnessvalid(iter)=feval(objfunc,xgbest,Xv,Yv,Hn);
        GL=100*(fitnessvalid(iter)./min(fitnessvalid)-1);%generalisation loss
        if GL>MaxGl 
            xgbest=Tlbest;
             gbest=TlG;
        end
TlG=gbest;
        Tlbest=xgbest;
        fit=feval(objfunc,pop,Xtr,Ytr,Hn);
        count=count+popsize;

        ind=(fit<=pbest);
        xpbest(ind,:)=pop(ind,:);
        pbest(ind)=fit(ind);
        [minf, indmin]=min(fit);
        if minf<gbest
            gbest=fit(indmin);
            xgbest=pop(indmin,:);
        end
        
            w=((Maxiter-iter1)/Maxiter)*(wMax-wMin)+wMin;
%         w=0.9;
        iter1=iter1+1;
        alpha0=0.5;
        alpha1=alpha0+iter/Maxiter;
        v=w.*v+c1*(rand*(xpbest-pop)+c2*rand*(repmat(xgbest,[popsize 1])-pop));
        ind=abs(v)>repmat(vmax,[popsize D]);
        v(ind)=sign(v(ind))*vmax(1);
        pop=pop+v;
        if iter>M
        fprintf('PSOruntime=%d Iteration=%d,  Best=%f\n',r,iter,gbest);
        M=M+20;
        end
%         disp(['PSO is training fuzynn (Iteration = ', num2str(iter),' ,MSE = ', num2str(gbest),')'])
        if count>Maxc
            break
        end
    end
    [gbest,xgbest]=BProp(xgbest,Xtr,Ytr,Xv,Yv,inp,Hn,5000);
    %%%Training Results
    [~,Yhtr]=feval(objfunc,xgbest,Xtr,Ytr,Hn);

    msetr=mean((Yhtr-Ytr).^2);
    disp(['MSE train=' num2str(msetr)])
    corrtr=corrcoef(Ytr,Yhtr);
    disp(['R2tr=' num2str(corrtr(2).^2)])
    %%%Test Results
    [~,Yht]=feval(objfunc,xgbest,Xt,Yt,Hn);

    msetest=mean((Yht-Yt).^2);
    
    %
    disp(['MSE test=' num2str(msetest)])
    corrt=corrcoef(Yt,Yht);
    disp(['R2test=' num2str(corrt(2).^2)])

    GCs(r,:)=[msetr,msetest,corrtr(2).^2,corrt(2).^2];
    
    
end

save('BP_PSO_NN_15','GCs')


