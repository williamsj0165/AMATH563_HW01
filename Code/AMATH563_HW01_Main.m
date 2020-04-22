%% Load the data

fprintf('\nLoading training data...\n\n')
trn_data = load('mnist_train.csv');
trn_labl = trn_data(:,1)';
trn_imag = trn_data(:,2:end);
fprintf('All training data loaded.\n')
% Matrix A_trn is trn_imag

fprintf('\n\n\n')
fprintf('Loading testing data...\n\n')
tst_data = load('mnist_test.csv');
tst_labl = tst_data(:,1);
tst_imag = tst_data(:,2:end);
fprintf('All testing data loaded.\n\n')
% Matrix A_tst is tst_imag


%% Form relevant matrices

N_trn = length(trn_labl);
N_tst = length(tst_labl);

% Creating the matrix B = [y1...yn] for training data
trn_labl_mtx = zeros(10,length(trn_labl));
for i = 1 : length(trn_labl)
    trn_labl_mtx(trn_labl(i)+1,i) = 1; % this arranges it 0, 1...8, 9
end
trn_labl_mtx = trn_labl_mtx';

% Creating the matrix B = [y1...yn] for testing data
tst_labl_mtx = zeros(10,length(tst_labl));
for i = 1 : length(tst_labl)
    tst_labl_mtx(tst_labl(i)+1,i) = 1; % this arranges it 0, 1...8, 9
end
tst_labl_mtx = tst_labl_mtx';

fprintf('Finished forming relevant matrices.\n\n')


%% Section III – Initial Solve and Initial Partitioning

% run K_Folds.m


%% Section IV - Partitioning by Digit

run A_trn_mtxs.m
run A_tst_mtxs.m

% Run only ONE of the following. Or else!!!

% run Digits_01.m
% run Digits_02.m
% run Digits_03.m
% run Digits_04.m

%% Section V - Advanced Solves

run Advanced_Algorithms.m