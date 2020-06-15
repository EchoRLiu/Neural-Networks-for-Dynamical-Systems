close all; clear all; clc

% Simulate KS system.
% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - u_xxxx,  periodic BCs 

rng(42);
N = 1024/8; training_input = []; training_output = [];
x = 32*pi*(1:N)'/N;
for i = 1:100
    u = cos(rand(1,1)*x/16).*(1+sin(rand(1,1)*x/16)); % Change to random result.
    v = fft(u);

    % % % % % %
    %Spatial grid and initial condition:
    h = 0.025;
    k = [0:N/2-1 0 -N/2+1:-1]'/16;
    L = k.^2 - k.^4;
    E = exp(h*L); E2 = exp(h*L/2);
    M = 16;
    r = exp(1i*pi*((1:M)-.5)/M);
    LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
    Q = h*real(mean( (exp(LR/2)-1)./LR ,2)); 
    f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)); 
    f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
    f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

    % Main time-stepping loop:
    uu = u; tt = 0;
    tmax = 100; nmax = round(tmax/h); nplt = floor((tmax/250)/h); g = -0.5i*k;

    for n = 1:nmax
        t = n*h;
        Nv = g.*fft(real(ifft(v)).^2);
        a = E2.*v + Q.*Nv;
        Na = g.*fft(real(ifft(a)).^2);
        b = E2.*v + Q.*Na;
        Nb = g.*fft(real(ifft(b)).^2);
        c = E2.*a + Q.*(2*Nb-Nv);
        Nc = g.*fft(real(ifft(c)).^2);
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; 
        if mod(n,nplt)==0
            u = real(ifft(v));
            uu = [uu,u]; tt = [tt,t]; 
        end
    end
    
    % Input & Output as training data.
    training_input = [training_input; uu(:,1:end-1)'];
    training_output = [training_output; uu(:,2:end)'];
    
%     close all;
%     % Plot results:
%     surf(tt,x,uu), shading interp, colormap(hot), axis tight 
%     % view([-90 90]), colormap(autumn); 
%     set(gca,'zlim',[-5 50]) 
%     save('kuramoto_sivishinky.mat','x','tt','uu')
%     figure(2), pcolor(x,tt,uu.'),shading interp, colormap(hot), axis off, pause(1.)
end
    
%%

% Seperate training and testing data.
% q = randperm(N);
% training_u = uu(q(1:1000),:); testing_u = uu(q(1001:end),:);

% % Input & Output as training data.
% training_input = []; training_output = [];
% figure(3)
% for i = 1:1000
%     y = training_u(i,:);
%     training_input = [training_input; y(1:end-1)'];
%     training_output = [training_output; y(2:end)'];
%     plot(tt, y, 'b', 0, y(1), 'ro'), hold on,
% end

%%

net = feedforwardnet([50 50 50]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net, training_input.', training_output.');

%% LSTM.

layers = [sequenceInputLayer(N) 
    lstmLayer(200, 'OutputMode', 'sequence')
    fullyConnectedLayer(50)
    dropoutLayer(.1)
    fullyConnectedLayer(N)
    regressionLayer];

options = trainingOptions('adam', 'MaxEpochs', 2000, 'MiniBatchSize', 5, ...
    'Plots', 'training-progress');

net = trainNetwork(training_input.', training_output.', layers, options);

%%

close all;
for i = 1:10
    
    u = cos(rand(1,1)*x/16).*(1+sin(rand(1,1)*x/16)); % Change to random result.
    v = fft(u);

    % % % % % %
    %Spatial grid and initial condition:
    h = 0.025;
    k = [0:N/2-1 0 -N/2+1:-1]'/16;
    L = k.^2 - k.^4;
    E = exp(h*L); E2 = exp(h*L/2);
    M = 16;
    r = exp(1i*pi*((1:M)-.5)/M);
    LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
    Q = h*real(mean( (exp(LR/2)-1)./LR ,2)); 
    f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)); 
    f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
    f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

    % Main time-stepping loop:
    uu = u; tt = 0;
    tmax = 100; nmax = round(tmax/h); nplt = floor((tmax/250)/h); g = -0.5i*k;

    for n = 1:nmax
        t = n*h;
        Nv = g.*fft(real(ifft(v)).^2);
        a = E2.*v + Q.*Nv;
        Na = g.*fft(real(ifft(a)).^2);
        b = E2.*v + Q.*Na;
        Nb = g.*fft(real(ifft(b)).^2);
        c = E2.*a + Q.*(2*Nb-Nv);
        Nc = g.*fft(real(ifft(c)).^2);
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; 
        if mod(n,nplt)==0
            u = real(ifft(v));
            uu = [uu,u]; tt = [tt,t]; 
        end
    end
    
    % Plot the difference btw the result of Real Trajectory and NN result.
    
    figure(2*i+2)
    pcolor(x,tt,uu.'),shading interp, colormap(hot), axis off, 
    title("Real Trajectory"),
    
    figure(2*i+3)
    uunn = zeros(N, 251); uunn(:,1) = u; x0 = u;
    for j = 2:length(tt)
        % y0 = net(x0);
        y0 = predict(net, x0, 'MiniBatchSize', 1);
        uunn(:,j) = y0; x0 = y0;
    end
    pcolor(x,tt,uunn.'),shading interp, colormap(hot), axis off, 
    title("Trajectory with NN"),
   
end

%% SVD the result.

[U_in, S_in, V_in] = svd(training_input.'); 
[U_out, S_out, V_out] = svd(training_output.'); 

figure(1)
plot(diag(S_in), 'ro','LineWidth',[2])
figure(2)
plot(diag(S_out), 'bo','LineWidth',[2])
% 60 ranks.

%%
rank = 100;
input =  training_input.' \U_in(:, 1:rank);
output = training_output.' \U_out(:, 1:rank);

%%

net = feedforwardnet([10 20 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net, input.', output.');

%%

close all; 
xx = 32*pi*(1:rank)'/rank;

for i = 1:10
    
    u = cos(rand(1,1)*x/16).*(1+sin(rand(1,1)*x/16)); % Change to random result.
    v = fft(u);

    % % % % % %
    %Spatial grid and initial condition:
    h = 0.025;
    k = [0:N/2-1 0 -N/2+1:-1]'/16;
    L = k.^2 - k.^4;
    E = exp(h*L); E2 = exp(h*L/2);
    M = 16;
    r = exp(1i*pi*((1:M)-.5)/M);
    LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
    Q = h*real(mean( (exp(LR/2)-1)./LR ,2)); 
    f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)); 
    f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
    f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

    % Main time-stepping loop:
    uu = u; tt = 0;
    tmax = 100; nmax = round(tmax/h); nplt = floor((tmax/250)/h); g = -0.5i*k;

    for n = 1:nmax
        t = n*h;
        Nv = g.*fft(real(ifft(v)).^2);
        a = E2.*v + Q.*Nv;
        Na = g.*fft(real(ifft(a)).^2);
        b = E2.*v + Q.*Na;
        Nb = g.*fft(real(ifft(b)).^2);
        c = E2.*a + Q.*(2*Nb-Nv);
        Nc = g.*fft(real(ifft(c)).^2);
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; 
        if mod(n,nplt)==0
            u = real(ifft(v));
            uu = [uu,u]; tt = [tt,t]; 
        end
    end
      
    % Plot the difference btw the result of Real Trajectory and NN result.
    
    % [u1, s1, v1] = svd(uu);
    
    figure(2*i+2)
    pcolor(xx,tt,(uu).'),shading interp, colormap(hot), axis off, 
    title("Real Trajectory"),
    
    figure(2*i+3)
    % ux0 = uu\U_in(:, 1:rank);
    uunn = zeros(rank, 251); x0 = (u\U_in(:, 1:rank)).'; uunn(:,1) = x0; 
    for j = 2:length(tt)
        y0 = net(x0);
        % y0 = predict(net, x0, 'MiniBatchSize', 1);
        uunn(:,j) = y0; x0 = y0;
    end
    % uunn = U_in(:, 1:rank) * uunn;
    pcolor(xx,tt,uunn.'),shading interp, colormap(hot), axis off, 
    title("Trajectory with NN"),
   
end


%%


close all; clear all; clc

t=0:0.05:10;
d1=0.1; d2=0.1; beta=1.0;
L=20; n=512; N=n*n;
x2=linspace(-L/2,L/2,n+1); x=x2(1:n); y=x;
kx=(2*pi/L)*[0:(n/2-1) -n/2:-1]; ky=kx;

% INITIAL CONDITIONS

[X,Y]=meshgrid(x,y);
[KX,KY]=meshgrid(kx,ky);
K2=KX.^2+KY.^2; K22=reshape(K2,N,1);

m=1; % number of spirals

u = zeros(length(x),length(y),length(t));
v = zeros(length(x),length(y),length(t));

u(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*cos(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));
v(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*sin(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));

% REACTION-DIFFUSION
uvt=[reshape(fft2(u(:,:,1)),1,N) reshape(fft2(v(:,:,1)),1,N)].';
[t,uvsol]=ode45('reaction_diffusion_rhs',t,uvt,[],K22,d1,d2,beta,n,N);


for j=1:length(t)-1
ut=reshape((uvsol(j,1:N).'),n,n);
vt=reshape((uvsol(j,(N+1):(2*N)).'),n,n);
u(:,:,j+1)=real(ifft2(ut));
v(:,:,j+1)=real(ifft2(vt));

figure(1)
pcolor(x,y,v(:,:,j+1)); shading interp; colormap(hot); colorbar; drawnow; 
end

save('reaction_diffusion_big.mat','t','x','y','u','v')

load reaction_diffusion_big
pcolor(x,y,u(:,:,end)); shading interp; colormap(hot)

%% Lorenz.

clear all, close all

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; 
% r=10; 
% r=28;
r=40;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];
for j=1:100  % training trajectories
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end
grid on, view(-23,18)

%%
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');


%%
figure(2)

% r = 40;
% r=17; 
r=35;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

x0=20*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])

figure(3)
subplot(3,2,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
subplot(3,2,3), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
subplot(3,2,5), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])


figure(2)
x0=20*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.'; x0=y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])

figure(3)
subplot(3,2,2), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
subplot(3,2,4), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
subplot(3,2,6), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])

%%
figure(2), view(-75,15)
figure(3)
subplot(3,2,1), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,2), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,3), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,4), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,5), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,6), set(gca,'Fontsize',[15],'Xlim',[0 8])
legend('Lorenz','NN')
