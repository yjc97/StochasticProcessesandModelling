clear, close all
I1 = audioread('people1.wav'); %,[1 5000]
I2 = audioread('hit_table.wav'); %,[1 50000]
I3 = audioread('music3.wav');
%I4 = audioread('people6.wav');
%x = linspace(-15000000, 15000000, 100000);
I4 = wgn(1, 132300, 1);


subplot(3,4,1),plot(I1),title('輸入信號1');
subplot(3,4,2),plot(I2),title('輸入信號2');
subplot(3,4,3),plot(I3),title('輸入信號3');
subplot(3,4,4),plot(I4),title('輸入信號4');


% 將其組成矩陣
II1 = I1';
II2 = I2';
II3 = I3';
II4 = I4;

S=[II1;II2;II3;II4];
Sweight=rand(size(S,1));

MixedS=Sweight*S;     % 將混合矩陣重新排列並輸出

subplot(3,4,5),plot(MixedS(1,:)),title('混合信號1');
subplot(3,4,6),plot(MixedS(2,:)),title('混合信號2');
subplot(3,4,7),plot(MixedS(3,:)),title('混合信號3');
subplot(3,4,8),plot(MixedS(4,:)),title('混合信號4');


% wavwrite(MixedS(1,:),8000,8,'1mixwav1.wav');%保存wav數據
% wavwrite(MixedS(1,:),8000,8,'1mixwav2.wav');
% wavwrite(MixedS(1,:),8000,8,'1mixwav3.wav');


MixedS_bak=MixedS;                  
%%%%%%%%%%%%%%%%%%%%%%%%%%  標准化  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(MixedS, 1);
m = size(MixedS, 2);
MixedS_mean=zeros(n,1);
for i=1:n
    MixedS_mean(i)=mean(MixedS(i,:)); %算每一橫行的平均
end                                        % 計算MixedS的均值

for i=1:n
    for j=1:m
        MixedS(i,j)=MixedS(i,j)-MixedS_mean(i);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%  白化  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MixedS_cov=cov(MixedS');                    % cov為求協方差的函數, 是一個m x m 的方振
[E,D]=eig(MixedS_cov);                      % 對信號矩陣的協方差函數進行特征值分解 , E是eigenvector orthogonal matrix, D 是 diagonal matrix of eigenvalue
Q= inv(sqrt(D)) *(E)';                        % Q為白化矩陣
MixedS_white=Q*MixedS;                      % MixedS_white為白化后的信號矩陣
IsI=cov(MixedS_white');                     % IsI應為單位陣            

%%%%%%%%%%%%%%%%%%%%%%%%　FASTICA算法  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=MixedS_white;                            % 以下算法將對X進行操作
[VariableNum,SampleNum]=size(X);
numofIC=VariableNum;                       % 在此應用中，獨立元個數等於變量個數
B=zeros(numofIC,VariableNum);              % 初始化列向量w的寄存矩陣,B=[b1  b2  ...   bd]
for r=1:numofIC
    i=1;maxIterationsNum=100000;               % 設置最大迭代次數（即對於每個獨立分量而言迭代均不超過此次數）
    IterationsNum=0;
    b=rand(numofIC,1)-0.5;                 % 隨機設置b初值
    b=b-B*B'*b;                            % 額外加
    b=b/norm(b);                           % 對b標准化 norm(b):向量元素平方和開根號
    while i<=maxIterationsNum+1
        if i == maxIterationsNum           % 循環結束處理
            fprintf('\n第%d分量在%d次迭代內並不收斂。', r,maxIterationsNum);
            break;
        end
        bOld=b;                          
        a2=1;
        a1 = 1;
        u=1;
        t=X'*b;
        g=t.*exp(-a2*t.^2/2);
        dg=(1-a2*t.^2).*exp(-a2*t.^2/2);
        b=((1-u)*t'*g*b + u*X*g)/SampleNum-mean(dg)*b;
       
        %g = tanh(a1*t);
        %dg = 1-(tanh(t)).^2;
        %b = ((1-u)*t'*g*b + u*X*g)/SampleNum-mean(dg)*b;
        
        %g = t.^3;
        %dg = 3*t.^2;
        %b = (X*g)/SampleNum - mean(dg)*b;
        
        
                                           % 核心公式
        b=b-B*B'*b;                        % 對b正交化
        b=b/norm(b); 
        if abs(abs(b'*bOld)-1)<1e-9        % 如果收斂，則
             B(:,r)=b;                     % 保存所得向量b
             break;
        end
        i=i+1;        
    end
    B(:,r)=b;                                % 保存所得向量b
end

%%%%%%%%%%%%%%%%%%%%%%%%%%  ICA計算的數據復原並構圖  %%%%%%%%%%%%%%%%%%%%%%%%%
ICAedS= B' * Q * MixedS_bak  ;                     % 計算ICA后的矩陣 乘Q

%a = [1];
%b = [1, -1];
%ICAedS(1,:) = filter(b, a, ICAedS(1,:)); 高通濾波 放大背景雜音

%a = [1];
%b = [1, 1, 1, 1, 1]/5;
%ICAedS(1,:) = filter(b, a, ICAedS(1,:)); 低通濾波 降低高頻音 凸顯低音

% 將混合矩陣重新排列並輸出
subplot(3,4,9),plot(ICAedS(1,:)),title('ICA解混信號1');
subplot(3,4,10),plot(ICAedS(2,:)),title('ICA解混信號2');
subplot(3,4,11),plot(ICAedS(3,:)),title('ICA解混信號3');
subplot(3,4,12),plot(ICAedS(4,:)),title('ICA解混信號4');


audiowrite('1.wav',ICAedS(1,:), 44100);%保存wav數據
audiowrite('2.wav',ICAedS(2,:), 44100);
audiowrite('3.wav',ICAedS(3,:), 44100);
audiowrite('4.wav',ICAedS(4,:), 44100);