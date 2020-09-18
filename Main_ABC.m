clear
close
clc
runtime=30;      % The number of test time
GCs=[0,0,0,0];
for r=1:runtime
    save ('temABC','r','runtime','GCs')
    clear
    load temABC
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
    runtime=1;
    Hn=15;%number of hidden nodes
    ww=0.0; %%%percent of replacement
    D=2*(inp+1)*Hn;
    lb=-10*ones(1,D);
    ub=10*ones(1,D);
    MaxGl=4;%percent

    %/* Control Parameters of ABC algorithm*/
    NP=40;
    Maxc=20000;
    FoodNumber=NP/2; %/*The number of food sources equals the half of the colony size*/
    
    % maxCycle=1000;
    count=0;
    %/* Problem specific variables*/
    maxCycle=1000;
    % D=30;
    % [lb,ub]=test_func_range(Func_ind,D);
    
    %     D=length(lb);
    objfunc='fitrbf1'; %cost function to be optimized
    
    
    % limit=300;
    limit=NP*D/2;
    dd=1;
    dd2=1;
    % runtime=1;%/*Algorithm can be run many times in order to see its robustness*/
    abc=zeros(1,maxCycle);
    fit=zeros(1,FoodNumber);
    
    

    
    for k=1:FoodNumber
        cent=rand(Hn,inp);
        %         [~,cent]=kmeans(Xtr,Hn,'distance','sqEuclidean',...
        %             'replicates',1,'start','uniform');
        centt=reshape(cent,1,inp*Hn);
        dist=max(max(pdist2(cent,cent)));
        sigma=(1.2)*ones(1,Hn);
        bias=rand;
        W(:,k)=0.2*rand((inp+1)*Hn,1)-0.1;
        temp=[W(:,k)',centt,sigma];

        Foods(k,:)=[W(:,k)',centt,sigma];
    end
    Fitness=feval(objfunc,Foods,Xtr,Ytr,Hn);
    
    %reset trial counters
    trial=zeros(1,FoodNumber);
    
    %/*The best food source is memorized*/
    BestInd=find(Fitness==min(Fitness));
    BestInd=BestInd(end);
    GlobalMin=Fitness(BestInd);
    GlobalParams=Foods(BestInd,:);
    %     wbest=W(:,BestInd);
    iter=1;
    gbest1=GlobalMin;
    M=1;
    while ((iter <= maxCycle)),
        Gp(iter,1)=GlobalMin;
        %         sol=zeros(1,D);
        %%%%%%%%% EMPLOYED BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%
        for i=1:(FoodNumber)
            
            
            sol=Foods(i,:);
            Param2Change=fix(rand(1,round(dd))*D)+1;
            %/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
            neighbour=fix(rand*(FoodNumber))+1;
            phi=(rand(1,round(dd))-0.5)*2;
            %/*Randomly selected solution must be different from the solution i*/
            while(neighbour==i)
                neighbour=fix(rand*(FoodNumber))+1;
            end;
            sol(Param2Change)=Foods(i,Param2Change)+(Foods(i,Param2Change)-Foods(neighbour,Param2Change)).*phi;
            
            
            FitnessSol=feval(objfunc,sol,Xtr,Ytr,Hn);
            count=count+1;

            % /*a Deb's rule is applied between the current solution i and its mutant*/
            
            if (FitnessSol<Fitness(i)) %/*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
                Foods(i,:)=sol;
                Fitness(i)=FitnessSol;
                %                 Fitness(i)=ObjValSol;
                trial(i)=0;
            else
                trial(i)=trial(i)+1;
            end
            
        end;
        %%%%%%%%%%%%%%%%%%%%%%%% CalculateProbabilities %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        for i=1:FoodNumber
            %             indlf=find(Foods(i,:)<lb);induf=find(Foods(i,:)>ub);
            %             sumindf=[indlf induf];
            if Fitness(i)>=0
                fit(i)=1/(1+Fitness(i));
            else
                fit(i)=1+abs(Fitness(i));
            end
            
            
            %                 prob(i)=(1-(sum(violation(i,:))./sum(sum(violation))))*0.5;
            
        end
        prob=fit./sum(fit);
        
        %%%%%%%%%%%%%%%%%%%%%%%% ONLOOKER BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %         Fitness=feval(objfunc,Foods,Func_ind);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        i=1;
        t=0;
        
        while(t<FoodNumber)
            if(rand<prob(i))
                t=t+1;
                
                %/*The parameter to be changed is determined randomly*/
                %                 Param2Change=fix(rand*D)+1;
                Param2Change=fix(rand(1,round(dd2))*D)+1;
                %/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
                neighbour=fix(rand*(FoodNumber))+1;
                
                %                     %/*Randomly selected solution must be different from the solution i*/
                while(neighbour==i)
                    neighbour=fix(rand*(FoodNumber))+1;
                end;
                
                sol=Foods(i,:);
                %                 phi=(rand-0.5)*2;
                phi=(rand(1,round(dd2))-0.5)*2;
                %  /*v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
                sol(Param2Change)=Foods(i,Param2Change)+(Foods(i,Param2Change)-Foods(neighbour,Param2Change)).*phi;

                
                %evaluate new solution

                FitnessSol=feval(objfunc,sol,Xtr,Ytr,Hn);
                count=count+1;
                if (FitnessSol<Fitness(i)) %/*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
                    Foods(i,:)=sol;
                    Fitness(i)=FitnessSol;
                    %                     Fitness(i)=ObjValSol;
                    trial(i)=0;
                else
                    trial(i)=trial(i)+1;
                end
                
            end;
            
            i=i+1;
            if (i==(FoodNumber)+1)
                i=1;
            end;
        end;
        
        %/*The best food source is memorized*/
        ind=find(Fitness==min(Fitness));
        ind=ind(end);
        if (Fitness(ind)<GlobalMin)
            
            GlobalMin=Fitness(ind);
            GlobalParams=Foods(ind,:);
            % wbest=W(:,ind);
        end
        %     [~,~,~,~,~,wbest]=fitrbf(GlobalParams,Xtr,Ytr,Hn);
        fitnessvalid(iter)=feval(objfunc,GlobalParams,Xv,Yv,Hn);
        GL=100*(fitnessvalid(iter)./min(fitnessvalid)-1);%generalisation loss
        if GL>MaxGl & iter>1
            GlobalParams=Tlbest;
            GlobalMin=TlG;
            %                     break
        end
        Tlbest=GlobalParams;
        TlG=GlobalMin;
        %%%%%%%%%%%% SCOUT BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %/*determine the food sources whose trial counter exceeds the "limit" value.
        %In Basic ABC, only one scout is allowed to occur in each cycle*/
        ind=find(trial==max(trial));
        ind=ind(end);
        %         ind=find(trial>limit);
        for ii=1:length(ind)
            if (trial(ind(ii))>limit)
                trial(ind(ii))=0;
                [~,cent]=kmeans(Xtr,Hn,'distance','sqEuclidean',...
                    'replicates',1,'start','uniform');
                centt=reshape(cent,1,inp*Hn);
                dist=max(max(pdist2(cent,cent)));
                sigma=(1.2)*ones(1,Hn);
                W(:,ind(ii))=0.2*rand((inp+1)*Hn,1)-0.1;
                Foods(ind(ii),:)=[centt,sigma];
                FitnessSol=feval(objfunc,Foods(ind(ii),:),Xtr,Ytr,Hn);
                count=count+1;
                Fitness(ind(ii))=FitnessSol;
                
            end;
        end;
        if iter>M
        fprintf('Runtime=%d Iter=%d Fitness=%g\n',r,iter,GlobalMin);
        M=M+20;
        end
        abc(iter)=GlobalMin;
        iter=iter+1;
        if count>Maxc
            break;
        end
        
    end % End of ABC
    [GlobalMin,GlobalParams]=BProp(GlobalParams,Xtr,Ytr,Xv,Yv,inp,Hn,5000);

    % toc
    %%%Training Results
    [~,Yhtr]=feval(objfunc,GlobalParams,Xtr,Ytr,Hn);
    % [~,Yhtr] = postdata(Xtr,Yhtr,xcent,xhalf,ycent,yhalf);
    % ind=Yhtr<0;
    % Yhtr(ind)=0;
    % [~,Ytr] = postdata(Xtr,Ytr,xcent,xhalf,ycent,yhalf);
    msetr=mean((Yhtr-Ytr).^2);
    disp(['MSE train=' num2str(msetr)])
    corrtr=corrcoef(Ytr,Yhtr);
    disp(['R2tr=' num2str(corrtr(2).^2)])
    %%%Test Results
    [~,Yht]=feval(objfunc,GlobalParams,Xt,Yt,Hn);
    % [~,Yht] = postdata(Xt,Yht,xcent,xhalf,ycent,yhalf);
    % ind=Yht<0;
    % Yht(ind)=0;
    % [~,Yt] = postdata(Xt,Yt,xcent,xhalf,ycent,yhalf);
    msetest=mean((Yht-Yt).^2);
    
    %
    disp(['MSE test=' num2str(msetest)])
    corrt=corrcoef(Yt,Yht);
    disp(['R2test=' num2str(corrt(2).^2)])

    GCs(r,:)=[msetr,msetest,corrtr(2).^2,corrt(2).^2];
end
save('BP_ABC_NN_15','GCs')

