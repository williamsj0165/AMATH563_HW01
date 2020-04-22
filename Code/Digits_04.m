%% 4.) training against only relevant; seeking b_pred_d; BSL and MPS

K_vect = [1 2 4 5 10 20 25 40 50 100 1000];

n_k_tst = 800;

err_dgt_4_bsl_k = zeros(length(K_vect),1);
err_dgt_4_mps_k = zeros(length(K_vect),1);
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
    X_dgt_4_bsl_tot= zeros(784,10);
    X_dgt_4_mps_tot = zeros(784,10);
    for k = 1 : K
        k
        for dgt = 1 : 10
            X_dgt_4_bsl = A_trn(c(dgt,k):d(dgt,k),:) \ ones(n_k,1);
            X_dgt_4_bsl_tot(:,dgt) = X_dgt_4_bsl_tot(:,dgt) + X_dgt_4_bsl;
            
            X_dgt_4_mps = pinv(A_trn(c(dgt,k):d(dgt,k),:)) * ones(n_k,1);
            X_dgt_4_mps_tot(:,dgt) = X_dgt_4_mps_tot(:,dgt) + X_dgt_4_mps;
        end
    end
    X_dgt_4_bsl = X_dgt_4_bsl_tot / K;
    X_dgt_4_mps = X_dgt_4_mps_tot / K;
    
    c_tst = zeros(10,1);
    d_tst = zeros(10,1);
    c_tst(1) = 1;
    d_tst(1) = n_k_tst;
    for dgt = 2 : 10
        c_tst(dgt) = c_tst(dgt-1) + dgt_count_tst(dgt-1);
        d_tst(dgt) = c_tst(dgt) + n_k_tst - 1;
    end
    c_tst
    d_tst
    
    c_chk = zeros(10,1);
    d_chk = zeros(10,1);
    c_chk(1) = 1;
    d_chk(1) = n_k_tst;
    for i = 2 : 10
        c_chk(i) = c_chk(i-1) + n_k_tst;
        d_chk(i) = c_chk(i) + n_k_tst - 1;
    end
    
    B_dgt_4_bsl = zeros(n_k_tst*10,10);
    B_dgt_4_mps = zeros(n_k_tst*10,10);
    for dgt = 1 : 10
        B_dgt_4_bsl(c_chk(dgt):d_chk(dgt),dgt) = A_tst(c_tst(dgt):d_tst(dgt),:) * X_dgt_4_bsl(:,dgt);
        B_dgt_4_mps(c_chk(dgt):d_chk(dgt),dgt) = A_tst(c_tst(dgt):d_tst(dgt),:) * X_dgt_4_mps(:,dgt);
    end
    
    preds_dgt_4_bsl = B_dgt_4_bsl;
    error_dgt_4_bsl = zeros(n_k_tst*10,1);
    for n_tst = 1 : n_k_tst*10
        [M,I] = max(preds_dgt_4_bsl(n_tst,:));
        preds_dgt_4_bsl(n_tst,:) = 0;
        preds_dgt_4_bsl(n_tst,I) = 1;

        % A perfect prediction is error(:) = 0;
        if isequal(preds_dgt_4_bsl(n_tst,:),tst_labl_mtx(n_tst,:))
            error_dgt_4_bsl(n_tst) = 0;
        else
            error_dgt_4_bsl(n_tst) = 1;
        end
    end
    err_dgt_4_bsl_k(K_iter) = 1 - sum(error_dgt_4_bsl)/(n_k_tst*10)
    
    preds_dgt_4_mps = B_dgt_4_mps;
    error_dgt_4_mps = zeros(n_k_tst*10,1);
    for n_tst = 1 : n_k_tst*10
        [M,I] = max(preds_dgt_4_mps(n_tst,:));
        preds_dgt_4_mps(n_tst,:) = 0;
        preds_dgt_4_mps(n_tst,I) = 1;

        % A perfect prediction is error(:) = 0;
        if isequal(preds_dgt_4_mps(n_tst,:),tst_labl_mtx(n_tst,:))
            error_dgt_4_mps(n_tst) = 0;
        else
            error_dgt_4_mps(n_tst) = 1;
        end
    end
    err_dgt_4_mps_k(K_iter) = 1 - sum(error_dgt_4_mps)/(n_k_tst*10)
end


%% 4.) Post-Processing

figure
semilogx(K_vect,err_dgt_4_bsl_k,'r.','Markersize',10)
hold on
semilogx(K_vect,err_dgt_4_mps_k,'b.','Markersize',10)
axis([0 max(K_vect) 0.09 0.11])
legend('Backslash','Pseudoinverse','location','northeast')
xlabel('k','fontsize',16)
ylabel('SR','fontsize',16)
ttl_str = sprintf('Success Rate vs. Number of Folds\nTraining, Testing against Same Digits');
title(ttl_str,'fontsize',20)
