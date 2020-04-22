%% Generating dgt_count

dgt_count_tst = zeros(10,1); % 0 through 9
for i = 1 : length(tst_labl)
    dgt = tst_labl(i);
    
    dgt_count_tst(dgt+1) = dgt_count_tst(dgt+1) + 1;
end


%% Initializing A_tst_d matrices

fprintf('Initializing A_tst_d matrices\n\n')
A_tst_0 = zeros(dgt_count_tst(0+1),784);
A_tst_1 = zeros(dgt_count_tst(1+1),784);
A_tst_2 = zeros(dgt_count_tst(2+1),784);
A_tst_3 = zeros(dgt_count_tst(3+1),784);
A_tst_4 = zeros(dgt_count_tst(4+1),784);
A_tst_5 = zeros(dgt_count_tst(5+1),784);
A_tst_6 = zeros(dgt_count_tst(6+1),784);
A_tst_7 = zeros(dgt_count_tst(7+1),784);
A_tst_8 = zeros(dgt_count_tst(8+1),784);
A_tst_9 = zeros(dgt_count_tst(9+1),784);


%% Generating A_tst_d matrices

fprintf('Populating A_tst_d matrices\n\n')
i = ones(10,1);
for n = 1 : N_tst
    dgt = tst_labl(n);
    
    if dgt == 0
        A_tst_0(i(0+1),:) = tst_imag(n,:);
        i(0+1) = i(0+1) + 1;
    
    elseif dgt == 1
        A_tst_1(i(1+1),:) = tst_imag(n,:);
        i(1+1) = i(1+1) + 1;
        
    elseif dgt == 2
        A_tst_2(i(2+1),:) = tst_imag(n,:);
        i(2+1) = i(2+1) + 1;
        
    elseif dgt == 3
        A_tst_3(i(3+1),:) = tst_imag(n,:);
        i(3+1) = i(3+1) + 1;
        
    elseif dgt == 4
        A_tst_4(i(4+1),:) = tst_imag(n,:);
        i(4+1) = i(4+1) + 1;
        
    elseif dgt == 5
        A_tst_5(i(5+1),:) = tst_imag(n,:);
        i(5+1) = i(5+1) + 1;
        
    elseif dgt == 6
        A_tst_6(i(6+1),:) = tst_imag(n,:);
        i(6+1) = i(6+1) + 1;
        
    elseif dgt == 7
        A_tst_7(i(7+1),:) = tst_imag(n,:);
        i(7+1) = i(7+1) + 1;
        
    elseif dgt == 8
        A_tst_8(i(8+1),:) = tst_imag(n,:);
        i(8+1) = i(8+1) + 1;
        
    elseif dgt == 9
        A_tst_9(i(9+1),:) = tst_imag(n,:);
        i(9+1) = i(9+1) + 1;
    end
end

A_tst = [ A_tst_0; A_tst_1; A_tst_2; A_tst_3; A_tst_4; A_tst_5; A_tst_6; A_tst_7; A_tst_8; A_tst_9 ];

