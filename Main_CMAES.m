% function cmaes3 cmaVersion = '3.40.beta';
%finalized at 12: 29 20/12/2014
% Author : Bahram Jafrasteh
% clc
% close all
% clear all
function Main_CMAES(Hn)
runtime=30;      % The number of test time
GCs=[0,0,0,0];
for r=1:runtime
    save ('temCMAES','r','runtime','GCs','Hn')
    clear
    load temCMAES
    rng('default')
    rng('shuffle')
    %/* Control Parameters of ABC algorithm*/
    % load esforid-partition
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
    %     Hn=15;%number of hidden nodes
    popsize=50;
    insigma=2;
    MaxGl=4;%percent
    objfunc='fitrbf1'; %cost function to be optimized
    Dim=2*(inp+1)*Hn;
    lb=-1*ones(1,Dim);
    ub=1*ones(1,Dim);
    for k=1:1
        %         [~,cent]=kmeans(Xtr,Hn,'distance','sqEuclidean',...
        %             'replicates',1,'start','uniform');
        cent=rand(Hn,inp);
        centt=reshape(cent,1,inp*Hn);
        dist=max(max(pdist2(cent,cent)));
        sigma=(1.2)*ones(1,Hn);
        %         bias=rand;
        W(:,k)=0.2*rand((inp+1)*Hn,1)-0.1;
        temp=[W(:,k)',centt,sigma];
        %                         [~,~,~,~,sait]=feval(objfunc,temp,Xtr,Ytr,Hn);
        %         W(:,k)=(Ytr'/sait')';%+20*rand-10;
        xstart(k,:)=[W(:,k)',centt,sigma];
    end
    xstart=xstart';
    
    % xstart=((ubounds-lbounds)*ones(1,popsize)).*rand(Dim,popsize)+lbounds*ones(1,popsize);
    % Input Defaults (obsolete, these are obligatory now)
    % Evaluate options
    stopFitness = -inf;
    stopMaxFunEvals =20000;
    stopMaxIter = 20000;
    stopFunEvals = inf;
    stopIter = inf;
    stopOnWarnings = 1;
    flgWarnOnEqualFunctionValues = 0;
    flgDiagonalOnly = 0;
    flgdisplay = 1;
    verbosemodulo = 1090;
    flgscience = 1;
    flgsaving = [];
    lbounds=-inf;
    ubounds=inf;
    % Options defaults: Other
    defopts.DiffMaxChange = inf; % maximal variable change(s), can be Nx1-vector';
    defopts.DiffMinChange = 0;   % minimal variable change(s), can be Nx1-vector';
    defopts.LBounds = '-Inf % lower bounds, scalar or Nx1-vector';
    defopts.UBounds = 'Inf  % upper bounds, scalar or Nx1-vector';
    defopts.EvalInitialX = 'yes  % evaluation of initial solution';
    defopts.CMA.cs = '(mueff+2)/(Dim+mueff+3)  % cumulation constant for step-size';
    defopts.CMA.damps = '1 + 2*max(0,sqrt((mueff-1)/(Dim+1))-1) + cs  % damping for step-size';
    defopts.CMA.ccum = '4/(Dim+4)  % cumulation constant for covariance matrix';
    defopts.CMA.ccov1 = '2 / ((Dim+1.3)^2+mueff)  % learning rate for rank-one update';
    defopts.CMA.ccovmu = '2 * (mueff-2+1/mueff) / ((Dim+2)^2+mueff) % learning rate for rank-mu update';
    defopts.Seed = 'sum(100*clock)  % evaluated if it is a string';
    
    % ---------------------- Handling Input Parameters ----------------------
    % Compose options opts
    opts = defopts;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Main
    counteval = 0; countevalNaN = 0;
    iter = 0;runtime=0;
    while iter <= runtime % for-loop does not work with resume
        iter = iter + 1;
        % ------------------------ Initialization -------------------------------
        %         xmean = mean((xstart), 2);
        xmean = xstart;
        lambda = popsize;
        %--------------------------------------------------------------
        % Do more checking and initialization
        flgresume=1;
        if flgresume % resume is on%
            %   xmean = mean(xstart, 2); % evaluate xstart again, because of iter
            maxdx = (opts.DiffMaxChange); % maximal sensible variable change
            mindx = (opts.DiffMinChange); % minimal sensible variable change
            
            % Initialize dynamic internal state parameters
            sigma = max(insigma);              % overall standard deviation
            pc = zeros(Dim,1); ps = zeros(Dim,1);  % evolution paths for C and sigma
            insigma = insigma * ones(Dim,1) ;
            diagD = insigma/max(insigma);      % diagonal matrix D defines the scaling
            diagC = diagD.^2;
            if flgDiagonalOnly ~= 1            % use at some point full covariance matrix
                B = eye(Dim,Dim);                      % B defines the coordinate system
                BD = B.*repmat(diagD',Dim,1);        % B*D for speed up only
                C = diag(diagC);                   % covariance matrix == BD*(BD)'
            end
            
            fitness.hist=NaN*ones(1,10+ceil(3*10*Dim/lambda)); % history of fitness values
            fitness.histsel=NaN*ones(1,10+ceil(3*10*Dim/lambda)); % history of fitness values
            fitness.histbest=[]; % history of fitness values
            fitness.histmedian=[]; % history of fitness values
            
            % Initialize boundary handling
            bnd.isactive = any(lbounds > -Inf) || any(ubounds < Inf);
            
            % ooo initial feval, for output only
            if iter == 1
                out.solutions.bestever.x = xmean;
                out.solutions.bestever.f = Inf;  % for simpler comparison below
                out.solutions.bestever.evals = counteval;
                bestever = out.solutions.bestever;
            end
            if 1
                fitness.hist(1)=feval(objfunc, xmean', Xtr,Ytr,Hn);
                
                fitness.histsel(1)=fitness.hist(1);
                counteval = counteval + 1;
                if fitness.hist(1) < out.solutions.bestever.f
                    out.solutions.bestever.x = xmean;
                    out.solutions.bestever.f = fitness.hist(1);
                    out.solutions.bestever.evals = counteval;
                    bestever = out.solutions.bestever;
                end
            else
                fitness.hist(1)=NaN;
                fitness.histsel(1)=NaN;
            end
            
            % initialize random number generator
            if ischar(opts.Seed)
                randn('state', eval(opts.Seed));     % random number generator state
            else
                randn('state', opts.Seed);
            end
            %qqq
            %  load(opts.SaveFilename, 'startseed');
            %  randn('state', startseed);
            %  disp(['SEED RELOADED FROM ' opts.SaveFilename]);
            startseed = randn('state');         % for retrieving in saved variables
            
            % Initialize further constants
            chiN=Dim^0.5*(1-1/(4*Dim)+1/(21*Dim^2));  % expectation of
            %   ||Dim(0,I)|| == norm(randn(Dim,1))
            
            countiter = 0;
            % Initialize records and output
            if iter == 1
                time.t0 = clock;
                
                % TODO: keep also median solution?
                out.evals = counteval;  % should be first entry
                out.stopflag = {};
                
                outiter = 0;
                
                % Write headers to output data files
                %     filenameprefix = opts.LogFilenamePrefix;
            end % iter == 1
        end % else flgresume
        % -------------------- Generation Loop --------------------------------
        stopflag = {};
        while isempty(stopflag)
            % set internal parameters
            if countiter == 0 || lambda ~= lambda_last
                lambda_hist = [countiter+1; lambda];
                lambda_last = lambda;
                % Strategy internal parameter setting: Selection
                mu = popsize/2; % number of parents/points for recombination
                weights = log(mu+0.5)-log(1:mu)'; % muXone array for weighted recombination
                mueff=sum(weights)^2/sum(weights.^2); % variance-effective size of mu
                weights = weights/sum(weights);     % normalize recombination weights array
                % Strategy internal parameter setting: Adaptation
                cc = eval(opts.CMA.ccum); % time constant for cumulation for covariance matrix
                cs = eval(opts.CMA.cs);
                %qqq cs = (mueff^0.5)/(Dim^0.5+mueff^0.5) % t-const for cumulation for step size control
                % new way
                %     if myevalbool(opts.CMA.on)
                ccov1 = eval(opts.CMA.ccov1);
                ccovmu = min(1-ccov1, eval(opts.CMA.ccovmu));
                damps = eval(opts.CMA.damps); %dsigma
                noiseReevals = 0; % more convenient in later coding
            end
            
            countiter = countiter + 1;
            % Generate and evaluate lambda offspring
            fitness.raw = repmat(NaN, 1, lambda + noiseReevals);
            fitness.raw(lambda + find(isnan(fitness.raw(1:noiseReevals)))) = NaN;
            for k=find(isnan(fitness.raw)),
                tries = 0;
                while isnan(fitness.raw(k))
                    arz(:,k) = randn(Dim,1); % resample
                    arx(:,k) = xmean + sigma * (BD * arz(:,k));                % Eq. (1)
                    if ~bnd.isactive
                        arxvalid(:,k) = arx(:,k);
                    else
                        temp=arx(:,k);
                        ind=temp<lbounds;
                        temp(ind)=lbounds(ind);
                        ind=temp>ubounds;
                        temp(ind)=ubounds(ind);
                        arxvalid(:,k)=temp;
                    end
                    fitness.raw(k) = feval(objfunc, arxvalid(:,k)', Xtr,Ytr,Hn);
                    
                    tries = tries + 1;
                    if isnan(fitness.raw(k))
                        countevalNaN = countevalNaN + 1;
                    end
                    if mod(tries, 100) == 0
                        warning([num2str(tries) ...
                            ' NaN objective function values at evaluation ' ...
                            num2str(counteval)]);
                    end
                end
                counteval = counteval + 1; % retries due to NaN are not counted
            end
            
            fitness.sel = fitness.raw;
            % Sort by fitness
            [fitness.raw, fitness.idx] = sort(fitness.raw);
            [fitness.sel, fitness.idxsel] = sort(fitness.sel);  % minimization
            fitness.hist(2:end) = fitness.hist(1:end-1);    % record short history of
            fitness.hist(1) = fitness.raw(1);               % best fitness values
            if length(fitness.histbest) < 120+ceil(30*Dim/lambda) || ...
                    (mod(countiter, 5) == 0  && length(fitness.histbest) < 2e4)  % 20 percent of 1e5 gen.
                fitness.histbest = [fitness.raw(1) fitness.histbest];          % best fitness values
                fitness.histmedian = [median(fitness.raw) fitness.histmedian]; % median fitness values
            else
                fitness.histbest(2:end) = fitness.histbest(1:end-1);
                fitness.histmedian(2:end) = fitness.histmedian(1:end-1);
                fitness.histbest(1) = fitness.raw(1);           % best fitness values
                fitness.histmedian(1) = median(fitness.raw);    % median fitness values
            end
            fitness.histsel(2:end) = fitness.histsel(1:end-1); % record short history of
            fitness.histsel(1) = fitness.sel(1);               % best sel fitness values
            
            % Calculate new xmean, this is selection and recombination
            xold = xmean; % for speed up of Eq. (2) and (3)
            xmean = arx(:,fitness.idxsel(1:mu))*weights;
            zmean = arz(:,fitness.idxsel(1:mu))*weights;%==D^-1*B'*(xmean-xold)/sigma
            %         if mu == 1
            fmean = fitness.sel(1);
            %         else
            %             fmean = NaN; % [] does not work in the latter assignment
            %         end
            
            % Cumulation: update evolution paths
            ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (B*zmean);          % Eq. (4)
            hsig = norm(ps)/sqrt(1-(1-cs)^(2*countiter))/chiN < 1.4 + 2/(Dim+1);
            %  hsig = norm(ps)/sqrt(1-(1-cs)^(2*countiter))/chiN < 1.5 + 1/(Dim-0.5);
            %  hsig = norm(ps) < 1.5 * sqrt(Dim);
            %  hsig = 1;
            
            pc = (1-cc)*pc ...
                + hsig*(sqrt(cc*(2-cc)*mueff)/sigma) * (xmean-xold);     % Eq. (2)
            % Adapt covariance matrix
            if ccov1 + ccovmu > 0                                                    % Eq. (3)
                artmp = arx(:,fitness.idxsel(1:mu))-repmat(xold,1,mu);
                C = (1-ccov1-ccovmu+(1-hsig)*ccov1*cc*(2-cc)) * C ... % regard old matrix
                    + ccov1 * pc*pc' ...     % plus rank one update
                    + ccovmu ...             % plus rank mu update
                    * sigma^-2 * artmp * (repmat(weights,1,Dim) .* artmp');
                diagC = diag(C);
            end
            
            % the following is de-preciated and will be removed in future
            % better setting for cc makes this hack obsolete
            if 11 < 2 && ~flgscience
                % remove momentum in ps, if ps is large and fitness is getting worse.
                % this should rarely happen.
                % this might very well be counterproductive in dynamic environments
                if sum(ps.^2)/Dim > 1.5 + 10*(2/Dim)^.5 && ...
                        fitness.histsel(1) > max(fitness.histsel(2:3))
                    ps = ps * sqrt(Dim*(1+max(0,log(sum(ps.^2)/Dim))) / sum(ps.^2));
                    if flgdisplay
                        disp(['Momentum in ps removed at [niter neval]=' ...
                            num2str([countiter counteval]) ']']);
                    end
                end
            end
            
            % Adapt sigma
            sigma = sigma * exp((norm(ps)/chiN - 1)*cs/damps);             % Eq. (5)
            
            if 11 < 3
                if countiter == 1
                    disp('*********** sigma set to const * ||x|| ******************');
                end
                sigma = 0.04 * mueff * sqrt(sum(xmean.^2)) / Dim; % 20D,lam=1000:25e3
                sigma = 0.3 * mueff * sqrt(sum(xmean.^2)) / Dim; % 20D,lam=(40,1000):17e3
            end
            
            % Update B and D from C
            
            if ~flgDiagonalOnly && (ccov1+ccovmu) > 0 && mod(countiter, 1/(ccov1+ccovmu)/Dim/10) < 1
                if ~isnan(C)
                    C=triu(C)+triu(C,1)'; % enforce symmetry to prevent complex numbers
                    [B,tmp] = eig(C);     % eigen decomposition, B==normalized eigenvectors
                    % effort: approx. 15*Dim matrix-vector multiplications
                    diagD = diag(tmp);
                    
                    if any(~isfinite(diagD))
                        clear idx; % prevents error under octave
                        save(['tmp' opts.SaveFilename]);
                        error(['function eig returned non-finited eigenvalues, cond(C)=' ...
                            num2str(cond(C)) ]);
                    end
                    if any(any(~isfinite(B)))
                        clear idx; % prevents error under octave
                        save(['tmp' opts.SaveFilename]);
                        error(['function eig returned non-finited eigenvectors, cond(C)=' ...
                            num2str(cond(C)) ]);
                    end
                    
                    % limit condition of C to 1e14 + 1
                    if min(diagD) <= 0
                        if stopOnWarnings
                            stopflag(end+1) = {'warnconditioncov'};
                        else
                            warning(['Iteration ' num2str(countiter) ...
                                ': Eigenvalue (smaller) zero']);
                            diagD(diagD<0) = 0;
                            tmp = max(diagD)/1e14;
                            C = C + tmp*eye(Dim,Dim); diagD = diagD + tmp*ones(Dim,1);
                        end
                    end
                    if max(diagD) > 1e14*min(diagD)
                        if stopOnWarnings
                            stopflag(end+1) = {'warnconditioncov'};
                        else
                            warning(['Iteration ' num2str(countiter) ': condition of C ' ...
                                'at upper limit' ]);
                            tmp = max(diagD)/1e14 - min(diagD);
                            C = C + tmp*eye(Dim,Dim); diagD = diagD + tmp*ones(Dim,1);
                        end
                    end
                    
                    diagC = diag(C);
                    diagD = sqrt(diagD); % D contains standard deviations now
                    % diagD = diagD / prod(diagD)^(1/Dim);  C = C / prod(diagD)^(2/Dim);
                    BD = B.*repmat(diagD',Dim,1); % O(n^2)
                end
            end % if mod
            
            % Align/rescale order of magnitude of scales of sigma and C for nicer output
            % not a very usual case
            if 1 < 2 && sigma > 1e10*max(diagD)
                fac = sigma / max(diagD);
                sigma = sigma/fac;
                pc = fac * pc;
                diagD = fac * diagD;
                if ~flgDiagonalOnly
                    C = fac^2 * C; % disp(fac);
                    BD = B.*repmat(diagD',Dim,1); % O(n^2), but repmat might be inefficient todo?
                end
                diagC = fac^2 * diagC;
            end
            
            if flgDiagonalOnly > 1 && countiter > flgDiagonalOnly
                % full covariance matrix from now on
                flgDiagonalOnly = 0;
                B = eye(Dim,Dim);
                BD = diag(diagD);
                C = diag(diagC); % is better, because correlations are spurious anyway
            end
            
            
            % ----- numerical error management -----
            % Adjust maximal coordinate axis deviations
            if any(sigma*sqrt(diagC) > maxdx)
                sigma = min(maxdx ./ sqrt(diagC));
            end
            % Adjust minimal coordinate axis deviations
            if any(sigma*sqrt(diagC) < mindx)
                sigma = max(mindx ./ sqrt(diagC)) * exp(0.05+cs/damps);
            end
            % Adjust too low coordinate axis deviations
            if any(xmean == xmean + 0.2*sigma*sqrt(diagC))
                if stopOnWarnings
                    stopflag(end+1) = {'warnnoeffectcoord'};
                else
                    warning(['Iteration ' num2str(countiter) ': coordinate axis std ' ...
                        'deviation too low' ]);
                    if flgDiagonalOnly
                        diagC = diagC + (ccov1_sep+ccovmu_sep) * (diagC .* ...
                            (xmean == xmean + 0.2*sigma*sqrt(diagC)));
                    else
                        C = C + (ccov1+ccovmu) * diag(diagC .* ...
                            (xmean == xmean + 0.2*sigma*sqrt(diagC)));
                    end
                    sigma = sigma * exp(0.05+cs/damps);
                end
            end
            % Adjust step size in case of (numerical) precision problem
            tmp = 0.1*sigma*BD(:,1+floor(mod(countiter,Dim)));
            if all(xmean == xmean + tmp)
                i = 1+floor(mod(countiter,Dim));
                if stopOnWarnings
                    stopflag(end+1) = {'warnnoeffectaxis'};
                else
                    warning(['Iteration ' num2str(countiter) ...
                        ': main axis standard deviation ' ...
                        num2str(sigma*diagD(i)) ' has no effect' ]);
                    sigma = sigma * exp(0.2+cs/damps);
                end
            end
            % Adjust step size in case of equal function values (flat fitness)
            if fitness.sel(1) == fitness.sel(1+ceil(0.1+lambda/4))
                if flgWarnOnEqualFunctionValues && stopOnWarnings
                    stopflag(end+1) = {'warnequalfunvals'};
                else
                    if flgWarnOnEqualFunctionValues
                        warning(['Iteration ' num2str(countiter) ...
                            ': equal function values f=' num2str(fitness.sel(1)) ...
                            ' at maximal main axis sigma ' ...
                            num2str(sigma*max(diagD))]);
                    end
                    sigma = sigma * exp(0.2+cs/damps);
                end
            end
            % Adjust step size in case of equal function values
            if countiter > 2 && (max([fitness.hist fitness.sel(1)])-min([fitness.hist fitness.sel(1)])) == 0
                if stopOnWarnings
                    stopflag(end+1) = {'warnequalfunvalhist'};
                else
                    warning(['Iteration ' num2str(countiter) ...
                        ': equal function values in history at maximal main ' ...
                        'axis sigma ' num2str(sigma*max(diagD))]);
                    sigma = sigma * exp(0.2+cs/damps);
                end
            end
            
            % ----- end numerical error management -----
            
            % Keep overall best solution
            out.evals = counteval;
            out.solutions.evals = counteval;
            out.solutions.mean.x = xmean;
            out.solutions.mean.f = fmean;
            out.solutions.mean.evals = counteval;
            out.solutions.recentbest.x = arxvalid(:, fitness.idx(1));
            out.solutions.recentbest.f = fitness.raw(1);
            out.solutions.recentbest.evals = counteval + fitness.idx(1) - lambda;
            out.solutions.recentworst.x = arxvalid(:, fitness.idx(end));
            out.solutions.recentworst.f = fitness.raw(end);
            out.solutions.recentworst.evals = counteval + fitness.idx(end) - lambda;
            if fitness.hist(1) < out.solutions.bestever.f
                out.solutions.bestever.x = arxvalid(:, fitness.idx(1));
                out.solutions.bestever.f = fitness.hist(1);
                out.solutions.bestever.evals = counteval + fitness.idx(1) - lambda;
                bestever = out.solutions.bestever;
            end
            
            
            if fitness.raw(1) <= stopFitness, stopflag(end+1) = {'fitness'}; end
            if counteval >= stopMaxFunEvals, stopflag(end+1) = {'maxfunevals'}; end
            if countiter >= stopMaxIter, stopflag(end+1) = {'maxiter'}; end
            
            
            if counteval >= stopFunEvals || countiter >= stopIter
                stopflag(end+1) = {'stoptoresume'};
                if length(stopflag) == 1 && flgsaving == 0
                    error('To resume later the saving option needs to be set');
                end
            end
            
            out.stopflag = stopflag;
            
            % ----- output generation -----
            if verbosemodulo > 0 && isfinite(verbosemodulo)
                if countiter == 1 || mod(countiter, 10*verbosemodulo) < 1
                    disp(['Iterat, #Fevals:   Function Value    (median,worst) ' ...
                        '|Axis Ratio|' ...
                        'idx:Min SD idx:Max SD']);
                end
                if mod(countiter, verbosemodulo) < 1 ...
                        || (verbosemodulo > 0 && isfinite(verbosemodulo) && ...
                        (countiter < 3 || ~isempty(stopflag)))
                    [minstd minstdidx] = min(sigma*sqrt(diagC));
                    [maxstd maxstdidx] = max(sigma*sqrt(diagC));
                    % format display nicely
                    disp([repmat(' ',1,4-floor(log10(countiter))) ...
                        num2str(countiter) ' , ' ...
                        repmat(' ',1,5-floor(log10(counteval))) ...
                        num2str(counteval) ' : ' ...
                        num2str(fitness.hist(1), '%.13e') ...
                        ' +(' num2str(median(fitness.raw)-fitness.hist(1), '%.0e ') ...
                        ',' num2str(max(fitness.raw)-fitness.hist(1), '%.0e ') ...
                        ') | ' ...
                        num2str(max(diagD)/min(diagD), '%4.2e') ' | ' ...
                        repmat(' ',1,1-floor(log10(minstdidx))) num2str(minstdidx) ':' ...
                        num2str(minstd, ' %.1e') ' ' ...
                        repmat(' ',1,1-floor(log10(maxstdidx))) num2str(maxstdidx) ':' ...
                        num2str(maxstd, ' %.1e')]);
                end
            end
            % save everything
            %   time.t3 = clock;
            if ~isempty(stopflag) || countiter == 100
                xmin = arxvalid(:, fitness.idx(1));
                fmin = fitness.raw(1);
            end
            %         [~,~,~,~,~,W]=fitrbf((out.solutions.bestever.x)',Xtr,Ytr,Hn);
            fitnessvalid(iter)=feval(objfunc,(out.solutions.bestever.x)',Xv,Yv,Hn);
            GL=100*(fitnessvalid(iter)./min(fitnessvalid)-1);%generalisation loss
            if GL>MaxGl
                fitness.raw(1)=Tlbest;
                arxvalid(:, fitness.idx(1))=TlG;
                %                         break
            end
            Tlbest=fitness.raw(1);
            TlG=arxvalid(:, fitness.idx(1));
            fprintf('Runtime=%d Counteval=%d ObjVal=%g\n',r,counteval,fitness.raw(1));
            % ----- end output generation -----
        end % while, end generation loop
        % -------------------- Final Procedures -------------------------------
        % Evaluate xmean and return best recent point in xmin
        fmin = fitness.raw(1);
        xmin = arxvalid(:, fitness.idx(1)); % Return best point of last generation.
        if length(stopflag) > sum(strcmp(stopflag, 'stoptoresume')) % final stopping
            out.solutions.mean.f = ...
                feval(objfunc, xmean', Xtr,Ytr,Hn);
            counteval = counteval + 1;
            out.solutions.mean.evals = counteval;
            if out.solutions.mean.f < fitness.raw(1)
                fmin = out.solutions.mean.f;
                %     xmin = xintobounds(xmean, lbounds, ubounds); % Return xmean as best point
                temp=xmean;
                ind=temp<lbounds;
                temp(ind)=lbounds(ind);
                ind=temp>ubounds;
                temp(ind)=ubounds(ind);
                xmin=temp;
            end
            if out.solutions.mean.f < out.solutions.bestever.f
                out.solutions.bestever = out.solutions.mean; % Return xmean as bestever point
                %             out.solutions.bestever.x = xintobounds(xmean, lbounds, ubounds);
                bestever = out.solutions.bestever;
            end
        end
        message = [];
        [out.solutions.bestever.f,out.solutions.bestever.x]=BProp(out.solutions.bestever.x',Xtr,Ytr,Xv,Yv,inp,Hn,5000);
        out.solutions.bestever.x=(out.solutions.bestever.x)';
        if flgdisplay
            disp(['#Fevals:   f(returned x)   |    bestever.f     | stopflag' ...
                message]);
            %   if isoctave
            strstop = stopflag(:);
            %   else
            %       strcat(stopflag(:), '.');
            %   end
            strstop = stopflag(:); %strcat(stopflag(:), '.');
            disp([repmat(' ',1,6-floor(log10(counteval))) ...
                num2str(counteval, '%6.0f') ': ' num2str(fmin, '%.11e') ' | ' ...
                num2str(out.solutions.bestever.f, '%.11e') ' | ' ...
                strstop{1:end}]);
            if Dim < 102
                disp(['min Fitness:' sprintf(' %+.1e', out.solutions.bestever.f)]);
            end
            if exist('sfile', 'var')
                disp(['Results saved in ' sfile]);
            end
        end
        
        out.arstopflags{iter} = stopflag;
        if any(strcmp(stopflag, 'fitness')) ...
                || any(strcmp(stopflag, 'maxfunevals')) ...
                || any(strcmp(stopflag, 'stoptoresume')) ...
                || any(strcmp(stopflag, 'manual'))
            break;
        end
    end % while iter <= runtime
    
    
    
    [~,Yhtr]=feval(objfunc,(out.solutions.bestever.x)',Xtr,Ytr,Hn);
    % [~,Yhtr] = postdata(Xtr,Yhtr,xcent,xhalf,ycent,yhalf);
    % ind=Yhtr<0;
    % Yhtr(ind)=0;
    % [~,Ytr] = postdata(Xtr,Ytr,xcent,xhalf,ycent,yhalf);
    msetr=mean((Yhtr-Ytr).^2);
    disp(['MSE train=' num2str(msetr)])
    corrtr=corrcoef(Ytr,Yhtr);
    disp(['R2tr=' num2str(corrtr(2).^2)])
    %%%Test Results
    [~,Yht]=feval(objfunc,(out.solutions.bestever.x)',Xt,Yt,Hn);
    % [~,Yht] = postdata(Xt,Yht,xcent,xhalf,ycent,yhalf);
    % ind=Yht<0;
    % Yht(ind)=0;
    % [~,Yt] = postdata(Xt,Yt,xcent,xhalf,ycent,yhalf);
    msetest=mean((Yht-Yt).^2);
    disp(['MSE test=' num2str(msetest)])
    corrt=corrcoef(Yt,Yht);
    disp(['R2test=' num2str(corrt(2).^2)])
    GCs(r,:)=[msetr,msetest,corrtr(2).^2,corrt(2).^2];
end
save(['BP_CMAES_NN_',num2str(Hn)],'GCs')

