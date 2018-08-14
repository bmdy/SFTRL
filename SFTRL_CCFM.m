%{
    min_{w, M}  
     1/n sum_{i=1}^n f(w'*a_i + 0.5 * a_i' * X * a_i) + lambda || M ||_*
%}

function [error, cost, BT_P, BT_N, norms, preds] = AFTRL_CCFM(data, model, opts)
    AT = data.A';
    b = data.b;
    [d, n] = size(AT);
    error = zeros(n, 1);
    errors = zeros(n, 1);
    cost = zeros(n, 1);
    norms = zeros(n,1);
    
    eta = opts.eta;
    m = opts.m;
    
    %g_w = zeros(d, 1);
    BT_P = zeros(d, 2*m); row_P = 0;
    BT_N = zeros(d, 2*m); row_N = 0;
    % X = B_P' * B_P - B_N' * B_N
    % f(w, X) = phi(w'*a_i + a_i' * X * a_i) * (w, X)
	count = 0;
    time_elapsed = 0;
    time = [];
    perf = [];
    
    fprintf('n=%d, d=%d, eta=%d\n', n, d, eta);
    tic;
    for t=1: floor(0.8*n)
        a = AT(:, t);
        %a = a/norm(a);
        BPa = BT_P' * a;
        BNa = BT_N' * a;
        sclar = BPa' * BPa - BNa' * BNa;
        z1 = model.grad(sclar * b(t)) * b(t);%for logistic
		%z1 = model.grad(sclar - b(t)) ;%for squared
        if t>1
            if sign(sclar) ~= b(t)
                error(t) = error(t-1) + 1;
            else
                error(t) = error(t-1);
            end
            errors(t) = errors(t-1) +  log(1+exp(-sclar*b(t)));
        end
        
        % update w
        %g_w = g_w + z1 * a;
        %w = -eta * g_w; 
        
        % update X = B_P' * B_P - B_N' * B_N
        if z1<=0
            row_P = row_P + 1;
            BT_P(:, row_P) =  sqrt(-eta * z1) * a;

            if (row_P==2*m)
                [U, S, ~] = svd(BT_P'*BT_P,'econ');
				count = count + 1;
                S(S<1e-12) = 0;
                rk = nnz(S);
                s = diag(S);
                V = BT_P * U(:, 1:rk) * (diag(1./ sqrt(s(1:rk))));

                if rk>=m
                    BT_P = [V(:,1:m-1) * diag(sqrt(s(1:m-1)-s(m))), zeros(d, m+1)];            
                    row_P = m-1; 
                else
                    BT_P = [V(:, 1:rk) * diag(sqrt(s(1:rk))), zeros(d, 2*m-rk)];
                    row_P = rk;
                end
            end
        else
            row_N = row_N + 1;
            BT_N(:, row_N) = sqrt(eta * z1) * a;

            if (row_N==2*m)
                [U, S, ~] = svd(BT_N'*BT_N,'econ');
				count = count + 1;
                S(S<1e-12) = 0;
                rk = nnz(S);
                s = diag(S);
                V = BT_N * U(:, 1:rk) * (diag(1./ sqrt(s(1:rk))));

                if rk>=m
                    BT_N = [V(:,1:m-1) * diag(sqrt(s(1:m-1)-s(m))), zeros(d, m+1)];            
                    row_N = m-1; 
                else
                    BT_N = [V(:, 1:rk) * diag(sqrt(s(1:rk))), zeros(d, 2*m-rk)];
                    row_N = rk;
                end
            end        
        end
        cost(t) = toc;
		norms(t) = nuclear_norm(BT_P, BT_N);
        if mod(t,floor(n/10))==0
            fprintf('%d/10 error=%f time=%f\n', t/floor(n/10), error(t)/t, cost(t));
        end
        if mod(t, 10) == 0
            time_elapsed = time_elapsed + toc;
            time = [time; time_elapsed];
            perf = [perf; errors(t)/t];
        end
    end
    fprintf('\n');
	display(count);
	test_error=zeros(floor(0.2*n), 1);
	preds = zeros(floor(0.2*n), 1);
	for t=floor(n*0.8)+1:n
		a = AT(:, t);
        BPa = BT_P' * a;
        BNa = BT_N' * a;
		sclar = BPa' * BPa - BNa' * BNa;
		preds(t-floor(n*0.8)) = (sclar);
        if t>floor(n*0.8)+1
            if sign(sclar) ~= b(t)
                test_error(t-floor(n*0.8)) = test_error(t-floor(n*0.8)-1) + 1;
            else
                test_error(t-floor(n*0.8)) = test_error(t-floor(n*0.8)-1);
            end
        end
	end
	display(test_error(floor(0.2*n))/floor(0.2*n));
    avg_loss_dir = strcat('E:\Researchs\OCFM/loss_ccfm_aftrl_spam');
    save(avg_loss_dir,'error');
    time_dir = strcat('E:\Researchs\OCFM/time_ccfm_aftrl_spam');
    save(time_dir,'time');
    perf_dir = strcat('E:\Researchs\OCFM/perf_ccfm_aftrl_spam');
    save(perf_dir,'perf');
end



