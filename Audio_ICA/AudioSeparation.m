clear, close all
I1 = audioread('people1.wav'); %,[1 5000]
I2 = audioread('hit_table.wav'); %,[1 50000]
I3 = audioread('music3.wav');
%I4 = audioread('people6.wav');
%x = linspace(-15000000, 15000000, 100000);
I4 = wgn(1, 132300, 1);


subplot(3,4,1),plot(I1),title('��J�H��1');
subplot(3,4,2),plot(I2),title('��J�H��2');
subplot(3,4,3),plot(I3),title('��J�H��3');
subplot(3,4,4),plot(I4),title('��J�H��4');


% �N��զ��x�}
II1 = I1';
II2 = I2';
II3 = I3';
II4 = I4;

S=[II1;II2;II3;II4];
Sweight=rand(size(S,1));

MixedS=Sweight*S;     % �N�V�X�x�}���s�ƦC�ÿ�X

subplot(3,4,5),plot(MixedS(1,:)),title('�V�X�H��1');
subplot(3,4,6),plot(MixedS(2,:)),title('�V�X�H��2');
subplot(3,4,7),plot(MixedS(3,:)),title('�V�X�H��3');
subplot(3,4,8),plot(MixedS(4,:)),title('�V�X�H��4');


% wavwrite(MixedS(1,:),8000,8,'1mixwav1.wav');%�O�swav�ƾ�
% wavwrite(MixedS(1,:),8000,8,'1mixwav2.wav');
% wavwrite(MixedS(1,:),8000,8,'1mixwav3.wav');


MixedS_bak=MixedS;                  
%%%%%%%%%%%%%%%%%%%%%%%%%%  �Э��  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(MixedS, 1);
m = size(MixedS, 2);
MixedS_mean=zeros(n,1);
for i=1:n
    MixedS_mean(i)=mean(MixedS(i,:)); %��C�@��檺����
end                                        % �p��MixedS������

for i=1:n
    for j=1:m
        MixedS(i,j)=MixedS(i,j)-MixedS_mean(i);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%  �դ�  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MixedS_cov=cov(MixedS');                    % cov���D���t�����, �O�@��m x m ���讶
[E,D]=eig(MixedS_cov);                      % ��H���x�}�����t��ƶi��S���Ȥ��� , E�Oeigenvector orthogonal matrix, D �O diagonal matrix of eigenvalue
Q= inv(sqrt(D)) *(E)';                        % Q���դƯx�}
MixedS_white=Q*MixedS;                      % MixedS_white���դƦZ���H���x�}
IsI=cov(MixedS_white');                     % IsI�������}            

%%%%%%%%%%%%%%%%%%%%%%%%�@FASTICA��k  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=MixedS_white;                            % �H�U��k�N��X�i��ާ@
[VariableNum,SampleNum]=size(X);
numofIC=VariableNum;                       % �b�����Τ��A�W�ߤ��ӼƵ����ܶq�Ӽ�
B=zeros(numofIC,VariableNum);              % ��l�ƦC�V�qw���H�s�x�},B=[b1  b2  ...   bd]
for r=1:numofIC
    i=1;maxIterationsNum=100000;               % �]�m�̤j���N���ơ]�Y���C�ӿW�ߤ��q�Ө����N�����W�L�����ơ^
    IterationsNum=0;
    b=rand(numofIC,1)-0.5;                 % �H���]�mb���
    b=b-B*B'*b;                            % �B�~�[
    b=b/norm(b);                           % ��b�Э�� norm(b):�V�q��������M�}�ڸ�
    while i<=maxIterationsNum+1
        if i == maxIterationsNum           % �`�������B�z
            fprintf('\n��%d���q�b%d�����N���ä����ġC', r,maxIterationsNum);
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
        
        
                                           % �֤ߤ���
        b=b-B*B'*b;                        % ��b�����
        b=b/norm(b); 
        if abs(abs(b'*bOld)-1)<1e-9        % �p�G���ġA�h
             B(:,r)=b;                     % �O�s�ұo�V�qb
             break;
        end
        i=i+1;        
    end
    B(:,r)=b;                                % �O�s�ұo�V�qb
end

%%%%%%%%%%%%%%%%%%%%%%%%%%  ICA�p�⪺�ƾڴ_��úc��  %%%%%%%%%%%%%%%%%%%%%%%%%
ICAedS= B' * Q * MixedS_bak  ;                     % �p��ICA�Z���x�} ��Q

%a = [1];
%b = [1, -1];
%ICAedS(1,:) = filter(b, a, ICAedS(1,:)); ���q�o�i ��j�I������

%a = [1];
%b = [1, 1, 1, 1, 1]/5;
%ICAedS(1,:) = filter(b, a, ICAedS(1,:)); �C�q�o�i ���C���W�� �Y��C��

% �N�V�X�x�}���s�ƦC�ÿ�X
subplot(3,4,9),plot(ICAedS(1,:)),title('ICA�ѲV�H��1');
subplot(3,4,10),plot(ICAedS(2,:)),title('ICA�ѲV�H��2');
subplot(3,4,11),plot(ICAedS(3,:)),title('ICA�ѲV�H��3');
subplot(3,4,12),plot(ICAedS(4,:)),title('ICA�ѲV�H��4');


audiowrite('1.wav',ICAedS(1,:), 44100);%�O�swav�ƾ�
audiowrite('2.wav',ICAedS(2,:), 44100);
audiowrite('3.wav',ICAedS(3,:), 44100);
audiowrite('4.wav',ICAedS(4,:), 44100);