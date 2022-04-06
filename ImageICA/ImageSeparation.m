clear, close all
a = imread('a.jpg');
b = imread('b.jpg');
c = imread('d.jpg');
a = imresize(a, [512, 512]);
b = imresize(b, [512, 512]);
c = imresize(c, [512, 512]);
d = rgb2gray(a);
e = rgb2gray(b);
f = rgb2gray(c);

I1 = reshape(d, [512*512, 1]);
I2 = reshape(e, [512*512, 1]);
I3 = reshape(f, [512*512, 1]);

subplot(3,3,1),imshow(d),title('input signal 1');
subplot(3,3,2),imshow(e),title('input signal 2');
subplot(3,3,3),imshow(f),title('input signal 3');


% 將其組成矩陣
II1 = I1';
II2 = I2';
II3 = I3';

S=[II1;II2;II3];
S = double(S);
Sweight=rand(size(S,1));      

MixedS=Sweight*S;     % 將混合矩陣重新排列並輸出

ms1 = reshape(MixedS(1,:), [512,512]);
ms2 = reshape(MixedS(2,:), [512,512]);
ms3 = reshape(MixedS(3,:), [512,512]);

%MI1 = unit8(round(ms1));
%MI2 = unit8(round(ms2));
%MI3 = unit8(round(ms3));


subplot(3,3,4),imshow(ms1),title('mixed signal 1');
subplot(3,3,5),imshow(ms2),title('mixed signal 2');
subplot(3,3,6),imshow(ms3),title('mixed signal 3');

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
    i=1;maxIterationsNum=10000;               % 設置最大迭代次數（即對於每個獨立分量而言迭代均不超過此次數）
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
        a1=1;
        a2=1;
        u=1;
        t=X'*b;
        
        g = tanh(a1*t);
        dg = 1-(tanh(t)).^2;
        b = ((1-u)*t'*g*b + u*X*g)/SampleNum-mean(dg)*b;
        
        %g=t.*exp(-a2*t.^2/2);
        %dg=(1-a2*t.^2).*exp(-a2*t.^2/2);
        %b=((1-u)*t'*g*b + u*X*g)/SampleNum-mean(dg)*b;
       
        %g = t.^3;
        %dg= 3*t.^2;
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
    fprintf('%f\n', i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%  ICA計算的數據復原並構圖  %%%%%%%%%%%%%%%%%%%%%%%%%
ICAedS= B' * Q * MixedS_bak  ;                     % 計算ICA后的矩陣 乘Q

Is1 = reshape(ICAedS(1,:), [512,512]);
Is2 = reshape(ICAedS(2,:), [512,512]);
Is3 = reshape(ICAedS(3,:), [512,512]);

%II1 = unit8(round(Is1));
%II2 = unit8(round(Is2));
%II3 = unit8(round(Is3));

% 將混合矩陣重新排列並輸出
subplot(3,3,7),imshow(Is1),title('ICA output signal 1');
subplot(3,3,8),imshow(Is2),title('ICA output signal 2');
subplot(3,3,9),imshow(Is3),title('ICA output signal 3');