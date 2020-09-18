%Ccukoo search algorithm with LM for
% training
% Author Bahram Jafrasteh
clear
clc
runtime=30;      % The number of test time
GCs=[0,0,0,0];
for r=1:runtime
    save ('temCS','r','runtime','GCs')
    clear
    load temCS
    rng('default')
    rng('shuffle')
    % load esforid-partition
    % load esfordi-partition2.mat
    % X=[X,new,new2];
%           load esfordi_for_LLWNGEo
load esfordi_LLRBF
    X=[X,new];
    %activation function
    pa=0.25;
    [X,~,xcent,xhalf,ycent,yhalf] = prepdata(X,Y);
    % load gridn
    [N,inp]=size(X);
    Xtr=X(1:size(Xtr,1),:);
    Ytr=Y(1:size(Xtr,1),:);
    Xv=X(size(Xtr,1)+1:size(Xtr,1)+size(Xv,1),:);
    Yv=Y(size(Xtr,1)+1:size(Xtr,1)+size(Xv,1));
    Xt=X(size(Xtr,1)+size(Xv,1)+1:size(Xtr,1)+size(Xv,1)+size(Xt,1),:);
    Yt=Y(size(Xtr,1)+size(Xv,1)+1:size(Xtr,1)+size(Xv,1)+size(Xt,1));
    nnest=40;% Number of nests (or different solutions)
    MaxIter=500;
    warning('off')
    objfunc='fitrbf1';
    [~,inp]=size(X);%number of membership function for each input
    Hn=15;%number of hidden nodes
    Dim=2*(inp+1)*Hn+1;
    change=1;
    Maxc=20000;
    MaxGl=4;%percent
    spc=-1:0.01:1;
    % lb=[-10*ones(1,Dim/2),-0.5*ones(1,Dim/2)];
    % ub=[10*ones(1,Dim/2),0.5*ones(1,Dim/2)];
    lb=-10*ones(1,Dim);
    ub=10*ones(1,Dim);
    for k=1:nnest
        cent=rand(Hn,inp);
        %     [~,cent]=kmeans(Xtr,Hn,'distance','cosine',...
        %         'replicates',5,'start','cluster');
        
        centt=reshape(cent,1,inp*Hn);
        dist=max(max(pdist2(cent,cent)));
        sigm=(1.2)*ones(1,Hn);
        W(:,k)=0.2*rand((inp+1)*Hn,1)-0.1;
        %     temp=[W(:,k)',centt,sigma];
        bias=rand;
        nest(k,:)=[W(:,k)',centt,sigm,bias];
        %      [~,~,~,~,~,W(:,k)]=feval(objfunc,nest(k,:),Xtr,Ytr,Hn);
    end

    fitn=feval(objfunc,nest,Xtr,Ytr,Hn);
    % end
    [~,inds]=sort(fitn);
    fitness=fitn(inds(1:nnest));
    % nest=tnest(inds(1:nnest),:);
    
    %Name of Optimization function
    %     objfunc='sphere';
    objfunc='fitrbf1'; %cost function to be optimized
    %% initialization
    %%%%time of calculation
    st=cputime;
    Cuckoo=zeros(1,1);
    
    
    %% ......................Get the current best ...................
    % Evaluating all new solutions
    % fitness=fitrbf(nest,Xtr,Ytr,Hn);
    % Find the current best
    [fmin,K]=min(fitness) ;
    bestnest=nest(K,:);
    % wbest=W(:,K);
    % [~,~,~,~,~,wbest]=fitrbf(bestnest,Xtr,Ytr,Hn);
    
    gbest1=fmin;
    %% Starting iterations
    iter=1;II=0;count=0;
    M=1;
    while (iter<MaxIter ),
        %     alfa=((MaxIter-iter)/MaxIter)^1*(alphamx-alphamn)+alphamn;
        % Get cuckoos by ramdom walk
        % Levy flights
        % Levy exponent and coefficient
        % For details, see equation (2.21), Page 16 (chapter 2) of the book
        % X. S. Yang, Nature-Inspired Metaheuristic Algorithms, 2nd Edition, Luniver Press, (2010).
        beta=3/2;
        sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
        u=randn(nnest,Dim)*sigma;
        v=randn(nnest,Dim);
        step=u./abs(v).^(1/beta);
        stepsize=0.01*step.*(nest-repmat(bestnest,[nnest 1]));
        %     stepsize=alfa*step.*(nest-repmat(bestnest,[nnest 1]));
        newnest=nest+stepsize.*randn(nnest,Dim);
%         ind=newnest<ones(nnest,1)*lb;
%         newnest(ind)=lb(1);
%         ind=newnest>ones(nnest,1)*ub;
%         newnest(ind)=ub(1);
        %% ..........................Find the current best nest...........................
        % Evaluating all new solutions
        %     fitnessn=feval(objfunc,newnest,Func_ind);%fitness of new nest
        fitnessn=feval(objfunc,newnest,Xtr,Ytr,Hn);
        count=count+nnest;
        ind=fitnessn<=fitness;
        fitness(ind)=fitnessn(ind);
        nest(ind,:)=newnest(ind,:);
        Kk=rand(size(nest))>pa;

        fitnessvalid(iter)=feval(objfunc,bestnest,Xv,Yv,Hn);
        %     fitnessvalid(iter)=fitrbf(bestnest,Xv,Yv,Hn,wbest);
        GL=100*(fitnessvalid(iter)./min(fitnessvalid)-1);%generalisation loss
        if GL>MaxGl
            bestnest=Tlbest;
            fmin=TlG;
        end
        Tlbest=bestnest;
TlG=fmin;
        %% Replace some nests by constructing new solutions/nests
        % A fraction of worse nests are discovered with a probability pa
        % n=size(nest,1);
        % Discovered or not -- a status vector
        %     for i=1:nnest
        %         Kk(i,:)= rand(1,size(nest,2))>prob(i);
        %     end
        %             Kk=rand(size(nest))>pa;
        
        % In the real world, if a cuckoo's egg is very similar to a host's eggs, then
        % this cuckoo's egg is less likely to be discovered, thus the fitness should
        % be related to the difference in solutions.  Therefore, it is a good idea
        % to do a random walk in a biased way with some random step sizes.
        % New solution by biased/selective random walks
        stepsize=rand*(nest(randperm(nnest),:)-nest(randperm(nnest),:));
        % stepsize=rand*(nest(randperm(nnest),:)-repmat(bestnest,[nnest,1]));
        newnest=nest+stepsize.*Kk;
%         ind=newnest<ones(nnest,1)*lb;
%         newnest(ind)=lb(1);
%         ind=newnest>ones(nnest,1)*ub;
%         newnest(ind)=ub(1);
        %% Evaluate this set of solutions
        % Find the current best nest
        %     fitnessn=feval(objfunc,newnest,Func_ind);
        fitnessn=feval(objfunc,newnest,Xtr,Ytr,Hn);
        count=count+nnest;
        ind=fitnessn<fitness;
        fitness(ind)=fitnessn(ind);
        nest(ind,:)=newnest(ind,:);
        

        
        % Find the current best
        [fminn,K]=min(fitness) ;%new fmin
        best=nest(K,:);
        if fminn<fmin,
            fmin=fminn;
            bestnest=best;
            
            %         [fmin,~,~,~,~,wbest]=fitrbf(bestnest,Xtr,Ytr,Hn);
        end
        Cuckoo(iter)=fmin;
        if iter>M
        fprintf('CSRuntime=%d Iter=%d ObjVal=%g\n',r,iter,fmin);
        M=M+20;
        end
        iter=iter+1;
        if count>Maxc
            break
        end
    end
    [fmin,bestnest]=BProp(bestnest,Xtr,Ytr,Xv,Yv,inp,Hn,5000);
    %End of iterations
    %% Display all the nests
    %%%Training Results
    [~,Yhtr]=feval(objfunc,bestnest,Xtr,Ytr,Hn);

    msetr=mean((Yhtr-Ytr).^2);
    disp(['MSE train=' num2str(msetr)])
    corrtr=corrcoef(Ytr,Yhtr);
    disp(['R2tr=' num2str(corrtr(2).^2)])
    %%%Test Results
    % [~,Yht]=fitrbf(bestnest,Xt,Yt,Hn,wbest);
    [~,Yht]=feval(objfunc,bestnest,Xt,Yt,Hn);

    msetest=mean((Yht-Yt).^2);
    disp(['MSE test=' num2str(msetest)])
    corrt=corrcoef(Yt,Yht);
    disp(['R2test=' num2str(corrt(2).^2)])
    
    GCs(r,:)=[msetr,msetest,corrtr(2).^2,corrt(2).^2];
end
save('BP_CS_NN_15','GCs')
