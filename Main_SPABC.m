%% Author Bahram Jafrasteh
%% Main_SPABC LLRBF NN
%%Last Modified May 9th 2014
%%%ALABC%%%
clc
clear
close
runtime=1;      % The number of test time
GCs=[0,0,0,0];
for r=1:runtime
    save ('temSPABC','r','runtime','GCs')
    clear all
    load temSPABC
    rng('default')
    rng('shuffle')
    %/* Control Parameters of ABC algorithm*/

load esfordi_LLRBF
X=[X,new];

    warning('off')
    [X,~,xcent,xhalf,ycent,yhalf] = prepdata(X,Y);
    [N,inp]=size(X);
    Xtr=X(1:size(Xtr,1),:);
    Ytr=Y(1:size(Xtr,1),:);
    Xv=X(size(Xtr,1)+1:size(Xtr,1)+size(Xv,1),:);
    Yv=Y(size(Xtr,1)+1:size(Xtr,1)+size(Xv,1));
    
    Xt=X(size(Xtr,1)+size(Xv,1)+1:size(Xtr,1)+size(Xv,1)+size(Xt,1),:);
    Yt=Y(size(Xtr,1)+size(Xv,1)+1:size(Xtr,1)+size(Xv,1)+size(Xt,1));

    count=0;
    Nc=0;
    
    NP=40;
    st = cputime;
    FoodNumber=NP/2; %/*The number of food sources equals the half of the colony size*/
    change=0;
    Maxc=20000;
    MaxBP=5000;
    Kmax=ones(1,FoodNumber);
    % SF=0.1;  %Scaling Factor
    maxCycle=Maxc/NP;
    suc1=0;
    suc2=0;
    %/* Problem specific variables*/
    Hn=20;%number of hidden nodes
    ww=0.0; %%%percent of replacement
    D=2*(inp+1)*Hn;
    limit=NP*D/2;
    lb=-8*ones(FoodNumber,D);
    ub=8*ones(FoodNumber,D);

    dd2=1;
    dd=1;
    MaxGl=4;%percent
    objfunc='fitrbf1'; %cost function to be optimized
    strategy=2*ones(1,FoodNumber);
    visualizationFlag=0;
    % cx=zeros(FoodNumber,D);
    % for runtime=1:1
    % counter=0;
    RABC_S=zeros(1,maxCycle);
    % K=300;
    ak=.001*ub(1)*(1.1-((1:maxCycle)./maxCycle));
    %     ak=40;
    ckl=0.001*ub(1)*(1.1-(1:maxCycle)./maxCycle);

    for k=1:FoodNumber
        %         cent=rand(Hn,inp);
       [~,cent]=kmeans(Xtr,Hn,'distance','cosine',...
            'replicates',5,'start','cluster');
        cent=rand(Hn,inp);
        centt=reshape(cent,1,inp*Hn);
%         dist=max(max(pdist2(cent,cent)));
        sigma=rand(1,Hn)+0.1;
        W(:,k)=.2*rand((inp+1)*Hn,1)-.1;
% temp=[W(:,k)',centt,sigma];
%              [~,~,~,~,sait]=feval(objfunc,temp,Xtr,Ytr,Hn);
%     W(:,k)=(Ytr'/sait')';
        bias=rand;
        temp=[W(:,k)',centt,sigma];

        Foods(k,:)=[W(:,k)',centt,sigma];
    end
                ind=Foods<lb;
            Foods(ind)=lb(ind);
            ind=Foods>ub;
            Foods(ind)=ub(ind);
    %     ObjVal=feval(objfunc,Foods,Hb,A,b,neqcstr,c);
    Fitness=feval(objfunc,Foods,Xtr,Ytr,Hn);
    %reset trial counters
    trial=zeros(1,FoodNumber);
    %/*The best food source is memorized*/
    [sortf,BestInd]=sort(Fitness);
    % sorts=[sortf ;BestInd];
    Bestind=BestInd(1);
    mb1=Foods(Bestind,:);
    ind2=FoodNumber-sum((sortf-min(Fitness))>0.01)+1;
    GlobalMin=Fitness(Bestind);
    GlobalParams=Foods(Bestind,:);
    % wbest=W(:,Bestind);
    MC=maxCycle;
    iter=1;
    Gp=zeros(iter,1);
    trs=zeros(MC,1);
    gbest1=GlobalMin;
    II=0;
    while ((iter <= maxCycle)),
        
        
        Gp(iter,1)=GlobalMin;%%%store previous value
        %%%%%%%%% EMPLOYED BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%
        sol=Foods;
        Param2Change=fix(rand(FoodNumber,round(dd))*D)+1;
        Param2Change2=fix(rand(FoodNumber,round(dd))*D)+1;
        
        neighbour1=fix(rand(1,FoodNumber)*FoodNumber)+1;
        neighbour2=fix(rand(1,FoodNumber)*FoodNumber)+1;
        in=1:FoodNumber;
        if FoodNumber>2
            while sum(neighbour1==in)>0 || sum(neighbour1==neighbour2)>0 ...
                    || sum(neighbour2==in)>0
                if sum(neighbour1==in)>0
                    neighbour1=fix(rand(1,FoodNumber)*FoodNumber)+1;
                elseif sum(neighbour2==in)>0
                    neighbour2=fix(rand(1,FoodNumber)*FoodNumber)+1;
                else
                    neighbour1=fix(rand(1,FoodNumber)*FoodNumber)+1;
                    neighbour2=fix(rand(1,FoodNumber)*FoodNumber)+1;
                end
            end
        end
        mincur=1e10;
        for i=1:FoodNumber
            dist=pdist2(Foods(i,:),Foods);
            [~,neighbour1]=max(dist);
            [mindist,iindsm]=min(dist);
            if mindist<mincur
                mincur=mindist;
                indsp=i;
                indsm=iindsm;
            end
            %             phi=(rand(1,round(dd))-0.5)*2;
            % j=randperm(FoodNumber,1);
            sol(i,Param2Change2(i,:))=Foods(i,Param2Change(i,:))+(Foods(i,Param2Change(i,:))-...
                Foods(neighbour1,Param2Change(i,:)))*2*cos(rand*2*pi);
        end
                        ind=sol<lb;
            sol(ind)=lb(ind);
            ind=sol>ub;
            sol(ind)=ub(ind);
        FitnessSol=feval(objfunc,sol,Xtr,Ytr,Hn);
        count=count+FoodNumber;
        indi=find(FitnessSol<Fitness);
        trial=trial+1;
        trial(indi)=0;
        Foods(indi,:)=sol(indi,:);
        Fitness(indi)=FitnessSol(indi);
        
        %%%%%%%%%%%%%%%%%%%%%%%% CalculateProbabilities %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%Tournament Selection

        
        a=zeros(1,FoodNumber);

        for i=1:FoodNumber
            for j=1:FoodNumber
                if Fitness(i)<=Fitness(j)
                    a(i)=a(i)+((Fitness(j)-Fitness(i))./Fitness(j));
                    %                     if Fitness(i)==Fitness(j)
                    %                         a(i)=a(i)+1;
                    %                     end
                end
            end
        end
        inda=(a==0);
        a(inda)=min(a);
        prob=a./sum(a);
        %         prob=mapminmax(prob,0.1,.5);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        i=1;
        t=1;
        jj=0;kk=0;
        
        while(t<=FoodNumber)
            if(rand<prob(i))
                t=t+1;
                sol=Foods(i,:);change=0;
%                 distr=pdist2(sol,GlobalParams);
%                 ck=distr./((sum(abs(sol))+1));
                sol1=sol;Fitness1=Fitness(i);
                sol2=sol1;
                %SPSA algorithm
                changet=0;
                sol1b=sol1;rr=0;
                dist=pdist2(Foods(i,:),Foods);
                dist(dist==0)=inf;
                [~,neighbour1]=min(dist);
                if strategy(i)==1
                    for kc=0:Nc
                        if changet==0;
                            gradt=0;
                            for kl=1:Kmax

                                delta=2*round(rand(1,D))-1;

                                solplus=sol-ckl(iter)*sign(delta);
                                solminus=sol+ckl(iter)*sign(delta);

                                Fsolp=feval(objfunc,solplus,Xtr,Ytr,Hn);
                                Fsolm=feval(objfunc,solminus,Xtr,Ytr,Hn);
                                count=count+2;
                                if Fsolp<Fitness1
                                    Fitness1=Fsolp;
                                    suc1=suc1+1;
                                    sol1b=solplus;change=1;
                                    so1=sol1b;
                                    Foods(i,:)=solplus;
                                    Fitness(i)=Fsolp;
                                elseif Fsolm<Fitness1
                                    Fitness1=Fsolm;
                                    sol1b=solminus;change=1;
                                    so1=sol1b;
                                    suc1=suc1+1;
                                    Foods(i,:)=solminus;
                                    Fitness(i)=Fsolm;
                                end
                                grad=(Fsolp-Fsolm)./(2*ckl(iter)*sign(delta));

                                gradt=gradt+grad;
                            end
                        end
                        Tsol=sol1;

                        sol1=sol1-ak(iter)*gradt;
% sol1=sol1-ck.*gradt;
                        if Kmax(i)==-1
                            sol1=sol1-ak(iter)*gradtemp{i};
% sol1=sol1-ck.*gradtemp{i};
                        end
                        % ak=0.08*(1-iter/maxCycle);
                        % sol1=sol1-ak*gradt;
                        % save temp
                        FitnessSol1=feval(objfunc,sol1,Xtr,Ytr,Hn);
                        count=count+1;
                        %                         ak=ak*rand;%exp(-iter/maxCycle);
                        if FitnessSol1<Fitness1;
                            Kmax(i)=-1;
                            gradtemp{i}=gradt;
                            change=1;changet=1;%ak=ak*rand;
                            sol1b=sol1;changed=0;
                            Foods(i,:)=sol1;
                            Fitness(i)=FitnessSol1;
                            Fitness1=FitnessSol1;
                            strategy(i)=1;
                            suc1=suc1+1;
                        else
                            changet=0;sol1=Tsol;
                            strategy(i)=2;
                            Kmax(i)=1;
                        end
                    end
                    if Nc<0
                        FitnessSol1=Fitness(i);
                    end
                elseif strategy(i)==2
                    Param2Change=round(unifrnd(1,D,1,dd2));

                    sol2(Param2Change)=Foods(i,Param2Change)+(cos(rand*2*pi))*...
                        (GlobalParams(Param2Change)-Foods(neighbour1,Param2Change));

                    Fitness2=feval(objfunc,sol2,Xtr,Ytr,Hn);
                    % Fitness1=inf;
                    count=count+1;
                    
                    if Fitness2<Fitness(i)
                        Foods(i,:)=sol2;
                        Fitness(i)=Fitness2;
                        strategy(i)=2;
                        suc2=suc2+1;
                    else
%                         if count>floor(Maxc/2)
                            strategy(i)=1;
%                         end
                    end

                end
                if change==0;
                    trial(i)=trial(i)+1;
                end
                
            end
            i=i+1;
            if (i==(FoodNumber)+1)
                i=1;
            end;
        end;
        %/*The best food source is memorized*/
        [sortf,BestInd]=sort(Fitness);
        % sorts=[sortf ;BestInd];
        Bestind=BestInd(1);
        [~,Bestind]=min(Fitness);
        if (Fitness(Bestind)<GlobalMin)
            GlobalMin=Fitness(Bestind);
            GlobalParams=Foods(Bestind,:);
            %             Nc=-1;
        else
            %           Nc=0;
        end
        fitnessvalid(iter)=feval(objfunc,GlobalParams,Xv,Yv,Hn);
        GL(iter)=100*(fitnessvalid(iter)./min(fitnessvalid)-1);%generalisation loss
        if GL(iter)>MaxGl %& iter>30 %& GlobalMin<25
%             if GL(iter-1)>MaxGl
%                 GlobalParams=Tlbest;
%                 break
%             end
GlobalParams=Tlbest;
GlobalMin=TlG;
        end
        Tlbest=GlobalParams;
        TlG=GlobalMin;
        %%%%%%%%%%%% SCOUT BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %/*determine the food sources whose trial counter exceeds the "limit" value.
        %In Basic ABC, only one scout is allowed to occur in each cycle*/
        ind=find(trial>limit);
        if length(ind)>1
            for ii=1:length(ind)
                if (trial(ind(ii))>limit)
                    trial(ind(ii))=0;
                    [~,cent]=kmeans(Xtr,Hn,'distance','sqEuclidean',...
                        'replicates',1,'start','uniform');
                    centt=reshape(cent,1,inp*Hn);
                    dist=max(max(pdist2(cent,cent)));
                    sigma=(1.2)*ones(1,Hn);
                    W(:,ind(ii))=0.2*rand((inp+1)*Hn,1)-0.1;
                    bias=rand;
                    Foods(ind(ii),:)=[W(:,ind(ii))',centt,sigma];
                    FitnessSol=feval(objfunc,Foods(ind(ii),:),Xtr,Ytr,Hn);
                    count=count+1;
                    Fitness(ind(ii))=FitnessSol;
                end;
                trs(iter)=trs(iter)+1;
            end
        end
        fprintf('Runtime=%d Iter=%d ObjVal=%g\n',r,iter,GlobalMin);
        iter=iter+1;
        if count>Maxc
            break
        end
        
    end % End of ABC
    [GlobalMin,GlobalParams]=BProp(GlobalParams,Xtr,Ytr,Xv,Yv,inp,Hn,MaxBP);

    % toc
    %%%Training Results
    [~,Yhtr]=feval(objfunc,GlobalParams,Xtr,Ytr,Hn);
%     ind=Yhtr<0;
%     Yhtr(ind)=0;
    msetr=mean((Yhtr-Ytr).^2);
    disp(['MSE train=' num2str(msetr)])
    corrtr=corrcoef(Ytr,Yhtr);
    disp(['R2tr=' num2str(corrtr(2).^2)])
    [~,Yht]=feval(objfunc,GlobalParams,Xt,Yt,Hn);
%         ind=Yht<0;
%     Yht(ind)=0;
    msetest=mean((Yht-Yt).^2);
    disp(['MSE test=' num2str(msetest)])
    corrt=corrcoef(Yt,Yht);
    disp(['R2test=' num2str(corrt(2).^2)])

    GCs(r,:)=[msetr,msetest,corrtr(2).^2,corrt(2).^2];
end
 save('BP_SPABC_NN_20','GCs')

