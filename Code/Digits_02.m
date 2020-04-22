%% 2.) training against only relevant images; seeking B_pred; BSL and MPS

K_vect = [1 2 4 5 10 20 25 40 50 100 1000];
% K_vect = [1 5 250]

err_dgt_2_bsl_k = zeros(length(K_vect),1);
err_dgt_2_mps_k = zeros(length(K_vect),1);
for K_iter = 1 : length(K_vect)
    K = K_vect(K_iter)
    n_k = 5000 / K;
    
    fprintf('Building X vectors\n\n')
    c = zeros(10,K);
    d = zeros(10,K);
    c(1,:) = 1;
    d(1,:) = n_k;
    for k = 1 : K
        for dgt = 2 : 10
            c(dgt,k) = dgt_count_trn(dgt-1) + c(dgt-1,k);
            d(dgt,k) = c(dgt,k) + n_k - 1;
        end
        c(:,k) = c(:,k) + n_k * (k-1);
        d(:,k) = d(:,k) + n_k * (k-1);
    end
    c
    d
    X_dgt_2_bsl_tot= zeros(784,10);
    X_dgt_2_mps_tot = zeros(784,10);
    for k = 1 : K
        k
        for dgt = 1 : 10
            X_dgt_2_bsl = A_trn(c(dgt,k):d(dgt,k),:) \ ones(n_k,1);
            X_dgt_2_bsl_tot(:,dgt) = X_dgt_2_bsl_tot(:,dgt) + X_dgt_2_bsl;
            
            X_dgt_2_mps = pinv(A_trn(c(dgt,k):d(dgt,k),:)) * ones(n_k,1);
            X_dgt_2_mps_tot(:,dgt) = X_dgt_2_mps_tot(:,dgt) + X_dgt_2_mps;
        end
    end
    X_dgt_2_bsl = X_dgt_2_bsl_tot / K;
    X_dgt_2_mps = X_dgt_2_mps_tot / K;

    B_dgt_2_bsl = tst_imag * X_dgt_2_bsl;
    preds_dgt_2_bsl = B_dgt_2_bsl;
    error_dgt_2_bsl = zeros(N_tst,1);
    for n_tst = 1 : N_tst
        [M,I] = max(preds_dgt_2_bsl(n_tst,:));
        preds_dgt_2_bsl(n_tst,:) = 0;
        preds_dgt_2_bsl(n_tst,I) = 1;

        % A perfect prediction is error(:) = 0;
        if isequal(preds_dgt_2_bsl(n_tst,:),tst_labl_mtx(n_tst,:))
            error_dgt_2_bsl(n_tst) = 0;
        else
            error_dgt_2_bsl(n_tst) = 1;
        end
    end
    err_dgt_2_bsl_k(K_iter) = 1 - sum(error_dgt_2_bsl)/N_tst

    
    B_dgt_2_mps = tst_imag * X_dgt_2_mps;
    preds_dgt_2_mps = B_dgt_2_mps;
    error_dgt_2_mps = zeros(N_tst,1);
    for n_tst = 1 : N_tst
        [M,I] = max(preds_dgt_2_mps(n_tst,:));
        preds_dgt_2_mps(n_tst,:) = 0;
        preds_dgt_2_mps(n_tst,I) = 1;

        % A perfect prediction is error(:) = 0;
        if isequal(preds_dgt_2_mps(n_tst,:),tst_labl_mtx(n_tst,:))
            error_dgt_2_mps(n_tst) = 0;
        else
            error_dgt_2_mps(n_tst) = 1;
        end
    end
    err_dgt_2_mps_k(K_iter) = 1 - sum(error_dgt_2_mps)/N_tst
    
end


%% 2.) Post-Processing

figure
semilogx(K_vect,err_dgt_2_bsl_k,'r.','Markersize',10)
hold on
semilogx(K_vect,err_dgt_2_mps_k,'b.','Markersize',10)
axis([0 max(K_vect) 0 1])
legend('Backslash','Pseudoinverse','location','northeast')
xlabel('k','fontsize',16)
ylabel('SR','fontsize',16)
ttl_str = sprintf('Success Rate vs. Number of Folds\nTraining against Same Digits, Testing against All Digits');
title(ttl_str,'fontsize',20)
