B_preds_las = zeros(N_tst,10);

lambda_vect = 0.01 : 0.01 : 0.2;
% lambda_vect = 0.1 : 0.1 : 1.0;
SR = zeros(length(lambda_vect),1);
for lambda_iter = 1 : length(lambda_vect)
    lambda = lambda_vect(lambda_iter)
    
    for dgt = 1 : 10
        [X,fitinfo] = lasso(trn_imag(1:N_trn,:),trn_labl_mtx(1:N_trn,dgt),'Lambda',lambda,'Alpha',0.001);
        B_preds_las(:,dgt) = tst_imag(1:N_tst,:) * X;
    end
    
    error_las = zeros(N_tst,1);
    for n_tst = 1 : N_tst
        [M,I] = max(B_preds_las(n_tst,:));
        B_preds_las(n_tst,:) = 0;
        B_preds_las(n_tst,I) = 1;
        % A perfect prediction is error(:) = 0;
        if isequal(B_preds_las(n_tst,:), tst_labl_mtx(n_tst,:))
            error_las(n_tst) = 0;
        else
            error_las(n_tst) = 1;
        end
        
    end
    
    SR(lambda_iter) = 1 - sum(error_las)/N_tst
    
end

fprintf('\nfinished looping thru lambdas\n\n')


%% Plotting

figure
plot(lambda_vect,SR,'r.','Markersize',20)
hold on
axis([0 max(lambda_vect) 0 1])
xlabel('\lambda','fontsize',16)
ylabel('SR','fontsize',16)
ttl_str = sprintf('Success Rate vs. Regularization\nElastic Net, Alpha = 0.5');
title(ttl_str,'fontsize',20)

