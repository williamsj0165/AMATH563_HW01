%% Folding over k loops

K_vect = [1 2 3 4 5 10 15 20 25 30 40 50 75 100 150 200 350 500 1000 2500 5000 10000]; % thank goodness 60000 has so many divisors

err_bsl_1_k = zeros(length(K_vect),1);
err_bsl_2_k = zeros(length(K_vect),1);
err_mps_1_k = zeros(length(K_vect),1);
err_mps_2_k = zeros(length(K_vect),1);
for K_iter = 1 : length(K_vect)
    K = K_vect(K_iter)
    n_k = N_trn / K;

    c = 1;
    d = n_k;    
    X_bsl_tot = zeros(784,10);
    X_mps_tot = zeros(784,10);
    for k = 1 : K
        k
        
        X_bsl = trn_imag(c:d,:) \ trn_labl_mtx(c:d,:);
        X_bsl_tot = X_bsl_tot + X_bsl;
        
        X_mps = pinv(trn_imag(c:d,:)) * trn_labl_mtx(c:d,:);
        X_mps_tot = X_mps_tot + X_mps;
        
        c = d+1;
        d = n_k*(k+1);
    end
    
    % Backslash
    X_bsl = X_bsl_tot / K;
    B_prd_bsl = tst_imag * X_bsl;
    preds_bsl_1 = B_prd_bsl;
    preds_bsl_2 = B_prd_bsl - 1;
    error_bsl_1 = zeros(N_tst,1);
    error_bsl_2 = zeros(N_tst,1);
    for n_tst = 1 : N_tst
        [M,I] = max(preds_bsl_1(n_tst,:));
        preds_bsl_1(n_tst,:) = 0;
        preds_bsl_1(n_tst,I) = 1;
        % A perfect prediction is error(:) = 0;
        if isequal(preds_bsl_1(n_tst,:),tst_labl_mtx(n_tst,:))
            error_bsl_1(n_tst) = 0;
        else
            error_bsl_1(n_tst) = 1;
        end
        
        [M,I] = min(preds_bsl_2(n_tst,:));
        preds_bsl_2(n_tst,:) = 0;
        preds_bsl_2(n_tst,I) = 1;
        % A perfect prediction is error(:) = 0;
        if isequal(preds_bsl_2(n_tst,:),tst_labl_mtx(n_tst,:))
            error_bsl_2(n_tst) = 0;
        else
            error_bsl_2(n_tst) = 1;
        end
    end
    
    err_bsl_1_k(K_iter) = 1 - sum(error_bsl_1)/N_tst
    err_bsl_2_k(K_iter) = 1 - sum(error_bsl_2)/N_tst
    
    % Pseudoinverse
    X_mps = X_mps_tot / K;
    B_prd_mps = tst_imag * X_mps;
    preds_mps_1 = B_prd_mps;
    preds_mps_2 = B_prd_mps - 1;
    error_mps_1 = zeros(N_tst,1);
    error_mps_2 = zeros(N_tst,1);
    for n_tst = 1 : N_tst
        [M,I] = max(preds_mps_1(n_tst,:));
        preds_mps_1(n_tst,:) = 0;
        preds_mps_1(n_tst,I) = 1;

        % A perfect prediction is error(:) = 0;
        if isequal(preds_mps_1(n_tst,:),tst_labl_mtx(n_tst,:))
            error_mps_1(n_tst) = 0;
        else
            error_mps_1(n_tst) = 1;
        end
        
        [M,I] = min(preds_mps_2(n_tst,:));
        preds_mps_2(n_tst,:) = 0;
        preds_mps_2(n_tst,I) = 1;
        % A perfect prediction is error(:) = 0;
        if isequal(preds_mps_2(n_tst,:),tst_labl_mtx(n_tst,:))
            error_mps_2(n_tst) = 0;
        else
            error_mps_2(n_tst) = 1;
        end
    end
    err_mps_1_k(K_iter) = 1 - sum(error_mps_1)/N_tst
    err_mps_2_k(K_iter) = 1 - sum(error_mps_2)/N_tst
    
end

%% Post-Processing

figure
semilogx(K_vect,err_bsl_1_k,'r.','Markersize',12)
hold on
% semilogx(K_vect,err_bsl_2_k,'ro','Markersize',8)
semilogx(K_vect,err_mps_1_k,'b.','Markersize',10)
% semilogx(K_vect,err_mps_2_k,'bo','Markersize',8)
axis([0 max(K_vect) 0 1])
legend('Backslash','Pseudoinverse','location','southwest')
xlabel('k','fontsize',16)
ylabel('SR','fontsize',16)
ttl_str = sprintf('Success Rate vs. Number of Folds k\nAveraging X_{train,k}');
title(ttl_str,'fontsize',20)

fprintf('Finished !\n\n')
