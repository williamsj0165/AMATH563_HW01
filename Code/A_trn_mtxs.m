%% Generating dgt_count

dgt_count_trn = zeros(10,1); % 0 through 9
for i = 1 : length(trn_labl)
    dgt = trn_labl(i);
    
    dgt_count_trn(dgt+1) = dgt_count_trn(dgt+1) + 1;
end


%% Initializing A_trn_d matrices

fprintf('Initializing A_trn_d matrices\n\n')
A_trn_0 = zeros(dgt_count_trn(0+1),784);
A_trn_1 = zeros(dgt_count_trn(1+1),784);
A_trn_2 = zeros(dgt_count_trn(2+1),784);
A_trn_3 = zeros(dgt_count_trn(3+1),784);
A_trn_4 = zeros(dgt_count_trn(4+1),784);
A_trn_5 = zeros(dgt_count_trn(5+1),784);
A_trn_6 = zeros(dgt_count_trn(6+1),784);
A_trn_7 = zeros(dgt_count_trn(7+1),784);
A_trn_8 = zeros(dgt_count_trn(8+1),784);
A_trn_9 = zeros(dgt_count_trn(9+1),784);


%% Generating A_trn_d matrices

fprintf('Populating A_trn_d matrices\n\n')
i = ones(10,1);
for n = 1 : N_trn
    dgt = trn_labl(n);
    
    if dgt == 0
        A_trn_0(i(0+1),:) = trn_imag(n,:);
        i(0+1) = i(0+1) + 1;
    
    elseif dgt == 1
        A_trn_1(i(1+1),:) = trn_imag(n,:);
        i(1+1) = i(1+1) + 1;
        
    elseif dgt == 2
        A_trn_2(i(2+1),:) = trn_imag(n,:);
        i(2+1) = i(2+1) + 1;
        
    elseif dgt == 3
        A_trn_3(i(3+1),:) = trn_imag(n,:);
        i(3+1) = i(3+1) + 1;
        
    elseif dgt == 4
        A_trn_4(i(4+1),:) = trn_imag(n,:);
        i(4+1) = i(4+1) + 1;
        
    elseif dgt == 5
        A_trn_5(i(5+1),:) = trn_imag(n,:);
        i(5+1) = i(5+1) + 1;
        
    elseif dgt == 6
        A_trn_6(i(6+1),:) = trn_imag(n,:);
        i(6+1) = i(6+1) + 1;
        
    elseif dgt == 7
        A_trn_7(i(7+1),:) = trn_imag(n,:);
        i(7+1) = i(7+1) + 1;
        
    elseif dgt == 8
        A_trn_8(i(8+1),:) = trn_imag(n,:);
        i(8+1) = i(8+1) + 1;
        
    elseif dgt == 9
        A_trn_9(i(9+1),:) = trn_imag(n,:);
        i(9+1) = i(9+1) + 1;
    end
end

A_trn = [ A_trn_0; A_trn_1; A_trn_2; A_trn_3; A_trn_4; A_trn_5; A_trn_6; A_trn_7; A_trn_8; A_trn_9 ];

