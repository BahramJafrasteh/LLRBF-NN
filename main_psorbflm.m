%%Local Linear wavelet neural network by pso algorithm
%Coding by Bahram Jafrasteh
%PhD student at IUT
%Date: 12/11/2014
%
%main
clc
close all
clear all
load esforid-partition
[X,Y,xcent,xhalf,ycent,yhalf] = prepdata(X,Y);
[N,inp]=size(X);
Xtr=X(1:659,:);
Ytr=Y(1:659,:);
Xv=X(660:759,:);
Yv=Y(660:759);
Xt=X(760:877,:);
Yt=Y(760:877);
Maxiter=85;
popsize=40;
[~,inp]=size(Xtr);%number of membership function for each input
Hn=15;%number of hidden nodes
Dim=(inp+1)*Hn;
wMax=0.9;
wMin=0.01;
change=1;
MaxGl=10;%percent

for k=1:popsize
    %     cent=rand(Hn,inp);
    [~,cent]=kmeans(Xtr,Hn,'distance','sqEuclidean',...
        'replicates',1,'start','uniform');
    %     cent=rand(Hn,inp);
    centt=reshape(cent,1,inp*Hn);
    dist=max(max(pdist2(cent,cent)));
    %     sigma=(dist/sqrt(2*Hn))*ones(1,Hn);
    sigma=(1.2)*ones(1,Hn);
    W(:,k)=0.1*rand((inp+1)*Hn,1);
    temp=[W(:,k)',centt,sigma]; 
    pop(k,:)=[centt,sigma]; 
%     [~,pop(k,:),W(:,k)] = trainrbf(Xtr,Ytr,pop(k,:),Hn,W(:,k),1);
%     [~,~,~,hi]=fitrbf(pop(k,:),Xtr,Ytr,Hn,W);
%     W=(Ytr'/hi')';
      
end

v=(0.2)*rand(popsize,Dim)-0.1;

%%
for i=1:popsize
fit(i)=fitrbf(pop(i,:),Xtr,Ytr,Hn,W(:,i));
end
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
figure(1)
gbest1=gbest;
II=0;iter1=1;
for iter=1:Maxiter
    if abs(gbest1-gbest)<0.001
        II=II+1;
        %
        %
        % %
    else
        gbest1=gbest;
        II=0;
    end
    
    if (II==40 & change==1) | (II==20 & change==0)

        II=0;
        v=0.5*rand(popsize,Dim)-1;
%                    [~,ind]=sort(fit);
        ind=randperm(popsize);
        %     selec=randi(popsize,2,1);
        for i=1:0
%             for ep=1:1
                [fitnew,popnew,tW] = trainrbf(Xtr,Ytr,pop(ind(i),:),Hn,W(:,ind(i)));
                if fitnew<gbest
                    fit(ind(i))=fitnew;
                    pop(ind(i),:)=popnew;
                    tW=reshape(tW,1,(inp+1)*Hn);
                    W(:,ind(i))=tW; 
                    gbest=fitnew;
                    xgbest=popnew;
                    wbest=tW;
                end
%             end
            %
            
        end
%         iter1=Maxiter;
%         if change
            [tgbest,txgbest,twbest] = trainrbf(X,Y,xgbest,Hn,wbest,1);
            if tgbest<gbest
                gbest=tgbest;
                xgbest=txgbest;
                wbest=twbest;
            end
            change=0;
%         end
        [minf, indmin]=min(fit);
        if minf<gbest
            gbest=fit(indmin);
            xgbest=pop(indmin,:);
        end
                for k=1:popsize
            %     cent=rand(Hn,inp);
            %     [~,cent]=kmeans(Xtr,Hn,'distance','sqEuclidean',...
            %         'replicates',1,'start','uniform');
            cent=rand(Hn,inp);
            centt=reshape(cent,1,inp*Hn);
            dist=max(max(pdist2(cent,cent)));
            %     sigma=(dist/sqrt(2*Hn))*ones(1,Hn);
            sigma=(1.2)*ones(1,Hn);
            W(:,k)=rand((inp+1)*Hn,1);
            temp=[W(:,k)',centt,sigma];
            pop(k,:)=[centt,sigma];
            %     [~,pop(k,:),W(:,k)] = trainrbf(Xtr,Ytr,pop(k,:),Hn,W(:,k),1);
            %     [~,~,~,hi]=fitrbf(pop(k,:),Xtr,Ytr,Hn,W);
            %     W=(Ytr'/hi')';
            
        end
    end
    %generalisation loss
    fitnessvalid(iter)=fitrbf(xgbest,Xv,Yv,Hn,wbest);
    GL=100*(fitnessvalid(iter)./min(fitnessvalid)-1);%generalisation loss
    if GL>MaxGl & iter>7
        xgbest=Tlbest;
        break
    end
    Tlbest=xgbest;
    for i=1:popsize
fit(i)=fitrbf(pop(i,:),Xtr,Ytr,Hn,W(:,i));
end
%     fit=fitrbf(pop,Xtr,Ytr,Hn,W);
    ind=(fit<=pbest);
    xpbest(ind,:)=pop(ind,:);
    pbest(ind)=fit(ind);
    [minf, indmin]=min(fit);
    if minf<gbest
        gbest=fit(indmin);
        xgbest=pop(indmin,:);
        wbest=W(:,indmin);
    end
    
    
    
    w=((Maxiter-iter1)/Maxiter)*(wMax-wMin)+wMin;
    iter1=iter1+1;
    alpha0=0.5;
    alpha1=alpha0+iter/Maxiter;
    v=w.*v+alpha1*(rand*(xpbest-pop)+rand*(repmat(xgbest,[popsize 1])-pop));
    ind=abs(v)>repmat(vmax,[popsize Dim]);
    v(ind)=sign(v(ind))*vmax(1);
    pop=pop+v;
    disp(['PSO is training fuzynn (Iteration = ', num2str(iter),' ,MSE = ', num2str(gbest),')'])
    temptemp=[temptemp gbest];
    alaki=fitrbf(xgbest,Xtr,Ytr,Hn,wbest);
    temptemp1=[temptemp1  alaki];
    plot(temptemp)
    hold on
    plot(temptemp1,'r')
    drawnow
end

%%%Training Results
[~,Yhtr]=fitrbf(xgbest,Xtr,Ytr,Hn,wbest);
[~,Yhtr] = postdata(Xtr,Yhtr,xcent,xhalf,ycent,yhalf);
ind=Yhtr<0;
Yhtr(ind)=0;
[~,Ytr] = postdata(Xtr,Ytr,xcent,xhalf,ycent,yhalf);
msetr=mean((Yhtr-Ytr).^2);
disp(['MSE train=' num2str(msetr)])
corrtr=corrcoef(Ytr,Yhtr);
disp(['R2tr=' num2str(corrtr(2).^2)])
%%%Test Results
[~,Yht]=fitrbf(xgbest,Xt,Yt,Hn,wbest);
[~,Yht] = postdata(Xt,Yht,xcent,xhalf,ycent,yhalf);
ind=Yht<0;
Yht(ind)=0;
[~,Yt] = postdata(Xt,Yt,xcent,xhalf,ycent,yhalf);
msetest=mean((Yht-Yt).^2);
disp(['MSE test=' num2str(msetest)])
corrt=corrcoef(Yt,Yht);
disp(['R2test=' num2str(corrt(2).^2)])


load gridimp

[~,aa]=fitrbf(xgbest,pps,pps(:,1),Hn,wbest);

% pps(ind,:)=[];
[pps2,aaa2] = postdata(pps,aa,xcent,xhalf,ycent,yhalf);
ind=aaa2<0;
aaa2(ind)=0.001;
aaa3=[pps2,aaa2];