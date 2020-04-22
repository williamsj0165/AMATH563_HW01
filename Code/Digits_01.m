%% 1.) training against all 60,000 images in k partitions; seeking B_pred; BSL and MPS

% K_vect = [1 2 3 4 5 10 15 20 25 30 40 50 75 100 125 150 175 200 500 1000]; % thank goodness 60000 has so many divisors
% K_vect = [1 5]
K_vect = [1];

err_dgt_1_bsl_k = zeros(length(K_vect),1);
err_dgt_1_mps_k = zeros(length(K_vect),1);
for K_iter = 1 : length(K_vect)
    K = K_vect(K_iter)
    n_k = N_trn / K;
    
    fprintf('Building X vectors\n\n')
    c = zeros(K,1);
    d = zeros(K,1);
    for k = 1 : K
        c(k) = 1 + n_k*(k-1);
        d(k) = n_k*k;
    end
    c
    d
    X_dgt_1_bsl_tot = zeros(784,10);
    X_dgt_1_mps_tot = zeros(784,10);
    for k = 1 : K
        k
        for dgt = 1 : 10
            X_dgt_1_bsl = trn_imag(c(k):d(k),:) \ trn_labl_mtx(c(k):d(k),dgt);
            X_dgt_1_bsl_tot(:,dgt) = X_dgt_1_bsl_tot(:,dgt) + X_dgt_1_bsl;
            
            X_dgt_1_mps = pinv(trn_imag(c(k):d(k),:)) * trn_labl_mtx(c(k):d(k),dgt);
            X_dgt_1_mps_tot(:,dgt) = X_dgt_1_mps_tot(:,dgt) + X_dgt_1_mps;
        end
    end
    X_dgt_1_bsl = X_dgt_1_bsl_tot / K;
    X_dgt_1_mps = X_dgt_1_mps_tot / K;
    
    B_dgt_1_bsl = tst_imag * X_dgt_1_bsl;
    preds_dgt_1_bsl = B_dgt_1_bsl;
    error_dgt_1_bsl = zeros(N_tst,1);
    for n_tst = 1 : N_tst
        [M,I] = max(preds_dgt_1_bsl(n_tst,:));
        preds_dgt_1_bsl(n_tst,:) = 0;
        preds_dgt_1_bsl(n_tst,I) = 1;

        % A perfect prediction is error(:) = 0;
        if isequal(preds_dgt_1_bsl(n_tst,:),tst_labl_mtx(n_tst,:))
            error_dgt_1_bsl(n_tst) = 0;
        else
            error_dgt_1_bsl(n_tst) = 1;
        end
    end
    err_dgt_1_bsl_k(K_iter) = 1 - sum(error_dgt_1_bsl)/N_tst

    B_dgt_1_mps = tst_imag * X_dgt_1_mps;
    preds_dgt_1_mps = B_dgt_1_mps;
    error_dgt_1_mps = zeros(N_tst,1);
    for n_tst = 1 : N_tst
        [M,I] = max(preds_dgt_1_mps(n_tst,:));
        preds_dgt_1_mps(n_tst,:) = 0;
        preds_dgt_1_mps(n_tst,I) = 1;

        % A perfect prediction is error(:) = 0;
        if isequal(preds_dgt_1_mps(n_tst,:),tst_labl_mtx(n_tst,:))
            error_dgt_1_mps(n_tst) = 0;
        else
            error_dgt_1_mps(n_tst) = 1;
        end
    end
    err_dgt_1_mps_k(K_iter) = 1 - sum(error_dgt_1_mps)/N_tst
    
end


%% 1.) Post-Processing

figure
semilogx(K_vect,err_dgt_1_bsl_k,'r.','Markersize',10)
hold on
semilogx(K_vect,err_dgt_1_mps_k,'b.','Markersize',10)
axis([0 max(K_vect) 0 1])
legend('Backslash','Pseudoinverse','location','southwest')
xlabel('k','fontsize',16)
ylabel('SR','fontsize',16)
ttl_str = sprintf('Success Rate vs. Number of Folds\nTraining, Testing against All Digits');
title(ttl_str,'fontsize',20)

