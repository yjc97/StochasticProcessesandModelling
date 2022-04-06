function [ICs,cub2]=my_fastica(cub,p);

    bnd=size(cub,3);
    xx=size(cub,1);
    yy=size(cub,2);

    x=reshape(cub,xx*yy,bnd);
    x=x'; 
    L=size(x,1);
    K=size(x,2);
    %====Sphereing =====
    u=mean(x,2); %dimension of u is 1*L
    x_hat=x-u*ones(1,K);
    m=mean(x,2);
    C=(x*x')/size(x,2)-m*m';

    [V,D]=eig(C);
    A=inv(sqrtm(D))*V';    % A is the whitening matrix....
    x_whitened=A*(x_hat);
    clear x; 
    clear x_hat;
    %====Sphering finished

    threshold = 0.00001;
    B=zeros(L);
    for round=1:p
        fprintf('IC %d', round);
        
        %===initial condition ===
        w=rand(L,1)-0.5;
        %===
        
        w=w-B*B'*w;
        w=w/norm(w);
        wOld=zeros(size(w));
        wOld2=zeros(size(w));
        i=1;
        
        while i<=1000
            w=w-B*B'*w;
            w=w/norm(w);
            if norm(w-wOld)<threshold || norm(w+wOld)<threshold
                B(:,round)=w;
                W(round,:)=w';
                break;
            end
            wOld2=wOld;
            wOld=w;
            w=(x_whitened*((x_whitened'*w).^3))/K-3*w;   % NegEntrophy 推倒出來的更新式, 利用 g(u)=u^3
            w=w/norm(w);
            i=i+1;
        end
        round=round+1;
        
    end

    ICs=W*x_whitened;
cub2=reshape(ICs',xx,yy,p);

%==========plot====

%figure;i=0;

%subplot(2,5,1),imagesc(cub2(:,:,i+1)),colormap(gray);
%subplot(2,5,2),imagesc(cub2(:,:,i+2)),colormap(gray);
%subplot(2,5,3),imagesc(cub2(:,:,i+3)),colormap(gray);
%subplot(2,5,4),imagesc(cub2(:,:,i+4)),colormap(gray);
%subplot(2,5,5),imagesc(cub2(:,:,i+5)),colormap(gray);
%subplot(2,5,6),imagesc(cub2(:,:,i+6)),colormap(gray);
%subplot(2,5,7),imagesc(cub2(:,:,i+7)),colormap(gray);
%subplot(2,5,8),imagesc(cub2(:,:,i+8)),colormap(gray);
%subplot(2,5,9),imagesc(cub2(:,:,i+9)),colormap(gray);
%subplot(2,5,10),imagesc(cub2(:,:,i+10)),colormap(gray);


