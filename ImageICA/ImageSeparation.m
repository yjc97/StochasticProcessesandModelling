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


% �N��զ��x�}
II1 = I1';
II2 = I2';
II3 = I3';

S=[II1;II2;II3];
S = double(S);
Sweight=rand(size(S,1));      

MixedS=Sweight*S;     % �N�V�X�x�}���s�ƦC�ÿ�X

ms1 = reshape(MixedS(1,:), [512,512]);
ms2 = reshape(MixedS(2,:), [512,512]);
ms3 = reshape(MixedS(3,:), [512,512]);

%MI1 = unit8(round(ms1));
%MI2 = unit8(round(ms2));
%MI3 = unit8(round(ms3));


subplot(3,3,4),imshow(ms1),title('mixed signal 1');
subplot(3,3,5),imshow(ms2),title('mixed signal 2');
subplot(3,3,6),imshow(ms3),title('mixed signal 3');

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
    i=1;maxIterationsNum=10000;               % �]�m�̤j���N���ơ]�Y���C�ӿW�ߤ��q�Ө����N�����W�L�����ơ^
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
    fprintf('%f\n', i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%  ICA�p�⪺�ƾڴ_��úc��  %%%%%%%%%%%%%%%%%%%%%%%%%
ICAedS= B' * Q * MixedS_bak  ;                     % �p��ICA�Z���x�} ��Q

Is1 = reshape(ICAedS(1,:), [512,512]);
Is2 = reshape(ICAedS(2,:), [512,512]);
Is3 = reshape(ICAedS(3,:), [512,512]);

%II1 = unit8(round(Is1));
%II2 = unit8(round(Is2));
%II3 = unit8(round(Is3));

% �N�V�X�x�}���s�ƦC�ÿ�X
subplot(3,3,7),imshow(Is1),title('ICA output signal 1');
subplot(3,3,8),imshow(Is2),title('ICA output signal 2');
subplot(3,3,9),imshow(Is3),title('ICA output signal 3');