% Author : Bahram Jafrasteh

clear all
clc

%Common Parameter Setting
runtime=30;      % The number of test time
GCs=[0,0,0,0];
for r=1:runtime
    save ('temDE','r','runtime','GCs')
    clear
    load temDE
    %        load esfordi_for_LLWNGEo
    load esfordi_LLRBF
    X=[X,new];
    rng('default')
    rng('shuffle')
    
    
    [X,~,xcent,xhalf,ycent,yhalf] = prepdata(X,Y);
    [N,inp]=size(X);
    Xtr=X(1:size(Xtr,1),:);
    Ytr=Y(1:size(Xtr,1),:);
    Xv=X(size(Xtr,1)+1:size(Xtr,1)+size(Xv,1),:);
    Yv=Y(size(Xtr,1)+1:size(Xtr,1)+size(Xv,1));
    Xt=X(size(Xtr,1)+size(Xv,1)+1:size(Xtr,1)+size(Xv,1)+size(Xt,1),:);
    Yt=Y(size(Xtr,1)+size(Xv,1)+1:size(Xtr,1)+size(Xv,1)+size(Xt,1));
    
    
    Popsize=60; 		% Populations size
    MF=0.4; 		% Mutation factor
    CR=0.9; 		% Crossover rate
    MaxIter=500; 	% Max iteration time
    [~,inp]=size(X);%number of membership function for each input
    Hn=15;%number of hidden nodes
    Dim=2*(inp+1)*Hn;
    change=1;
    
    fueval=1;
    MaxGl=4;%percent
    Maxc=20000;
    objfunc='fitrbf1'; %cost function to be optimized
    spc=-1:0.01:1;
    
    for k=1:Popsize
        cent=rand(Hn,inp);
        %     [~,cent]=kmeans(Xtr,Hn,'distance','cosine',...
        %         'replicates',5,'start','cluster');
        
        centt=reshape(cent,1,inp*Hn);
        dist=max(max(pdist2(cent,cent)));
        sigma=(1.2)*ones(1,Hn);
        W(:,k)=0.2*rand((inp+1)*Hn,1)-0.1;
        bias=rand;
        temp=[W(:,k)',centt,sigma];
        XY(k,:)=[W(:,k)',centt,sigma];
        %      [~,~,~,~,~,W(:,k)]=feval(objfun,Foods(k,:),Xtr,Ytr,Hn);
    end
    FitX=feval(objfunc,XY,Xtr,Ytr,Hn);
    
    %     [~,inds]=sort(fitn);
    %     FitX=fitn(inds(1:Popsize));
    %     X=tX(inds(1:Popsize),:);
    %             if tYmin<Ymin
    [Ymin,ind]=min(FitX);
    bestnest=XY(ind,:);
    %         end
    iter=1;
    gbest1=min(FitX);
    II=0;
    count=0;par=0;
    M=1;
    for iter=1:MaxIter  % Stop when the iteration large than the max iteration time
        
        %             iter=iter+1;
        for m=1:Popsize % For each individual
            % Mutation
            R=randperm(Popsize);
            j=R(1);
            k=R(2);
            p=R(3);
            u=R(4);
            v=R(5);
            if j==m
                j=R(6);
            elseif k==m
                k=R(6);
            elseif p==m
                p=R(6);
            elseif u==m
                u=R(6);
            elseif v==m
                v=R(6);
            end
            V=XY(j,:)+MF*(XY(k,:)-XY(p,:));
            
            
            % Crossover put the result in the U matrix
            jrand=floor(rand()*Dim+1);
            for n=1:Dim
                R1=rand();
                if (R1<CR || n==jrand)
                    U(1,n)=V(1,n);
                else
                    U(1,n)=XY(m,n);
                end
            end
            
            %             FitU=feval(objfunc,U,Xtr,Ytr,Hnf);
            FitU=feval(objfunc,U,Xtr,Ytr,Hn);
            count=count+1;
            % Use the selection result to replace the m row
            % Selection
            if FitU < FitX(m)
                FitX(m)=FitU;
                XY(m,:)=U;
            end
            
            
            
            % Evaluate each individual's fitness value, and put the result in the FitX matrix.
            
            %                 FitX(m,1)=feval(objfunc,X(m,:),Func_ind);
        end % Now the 1th individual generated
        
        % Select the lowest fitness value
        [y,ind1]=sort(FitX,1);
        Y_min=y(1,1);
        [tYmin,ind] = min(FitX);
        if tYmin<Ymin
            Ymin=tYmin;
            bestnest=XY(ind,:);
        end
        
        fitnessvalid(iter)=feval(objfunc,bestnest,Xv,Yv,Hn);
        GL=100*(fitnessvalid(iter)./min(fitnessvalid)-1);%generalisation loss
        if GL>MaxGl
            
            bestnest=Tlbest;
            Ymin=TlG;
            %             break
        end
        Tlbest=bestnest;
        TlG=Ymin;
        if iter>M
            fprintf('DELMruntime=%d Iteration=%d,  Best=%f\n',r,iter,Ymin);
            M=M+20;
        end
        if count>Maxc
            break
        end
    end % Finish MaxIter times iteration
    [Ymin,bestnest]=BProp(bestnest,Xtr,Ytr,Xv,Yv,inp,Hn,5000);
    %     [minlm,indmlm]=min(fminlm);
    %     if minlm<Ymin
    %         Ymin=minlm;
    %         bestnest=bestnestlm(indmlm,:);
    %     end
    %%%Training Results
    [~,Yhtr]=feval(objfunc,bestnest,Xtr,Ytr,Hn);
    
    msetr=mean((Yhtr-Ytr).^2);
    disp(['MSE train=' num2str(msetr)])
    corrtr=corrcoef(Ytr,Yhtr);
    disp(['R2tr=' num2str(corrtr(2).^2)])
    %%%Test Results
    [~,Yht]=feval(objfunc,bestnest,Xt,Yt,Hn);
    
    msetest=mean((Yht-Yt).^2);
    disp(['MSE test=' num2str(msetest)])
    corrt=corrcoef(Yt,Yht);
    disp(['R2test=' num2str(corrt(2).^2)])
    
    GCs(r,:)=[msetr,msetest,corrtr(2).^2,corrt(2).^2];
end % Run 30 times
save('BP_DE_NN_15','GCs')
