% ------------------------------------------------------------------------
% PostGrad Course : Digital Signal Processing (2)
% Author: Muhammad Ewais
% Task: Checking new variable step size NLMS approach
% Last change: 2022-12-26
%% ------------------------------------------------------------------------ 
clear all;
close all;

% The goal of the adaptive noise cancellation is to estimate the desired
% signal d(n) from a noise-corrupted observation x(n) = d(n) + v1(n)
%% Adaptive filter
%N = 10000; % number of samples
N = 400; % number of samples
R = 100; % ensemble average over 100 runs
%nord = 12; % filter order
nord = 128; % filter order
% NLMS coefficients
%%beta = 0.25; % normalized step size
%beta = 0.05; % normalized step size
mu_nlms = 0.002; % normalized step size
% KWONG-NLMS coefficients
%alpha_kwong = 0.997;
alpha_kwong = 0.89;
gamma_kwong = 0.018;

% NLMS-Proposed coefficients
alpha_new=0.25;
beta_new=0.999;
gamma_new=0.02;
p_new=1;

% create arrays of all zeros
MSE_NLMS = zeros(N,1);
MSE_KWONG = zeros(N,1);
MSE_NEW = zeros(N,1);

dhat_NLMS = zeros(N,1);
dhat_KWONG = zeros(N,1);
dhat_NEW = zeros(N,1);

err_NLMS = zeros(N,1);
err_KWONG = zeros(N,1);
err_NEW = zeros(N,1);

add_figures=0;
if (add_figures) 
disp('Figures will be shown');
else
disp('Figures will not be shown');    
end

for r=1:R % used for computation of learning curves
    fprintf('Processing iteration number %d out of %d.\n',r,R);

%% Noise-corrupted observation: x(n) = d(n) + v1(n)
d = sin([1:N]*0.05*pi); % desired signal
g = randn(1,N)*0.25; % Gaussian white noise with a variance of 0.25
%v1= filter(1,[1 -0.8],g); % filtered white noise
v1= filter(1,[1 -0.8],g); % filtered white noise
x = d + v1;

% figure(100)
% plot(g(1:1000))
% figure(101)
% plot(d(1:1000))


if (add_figures) 
% plot of d(n) and x(n)
figure(1)
subplot(2,1,1)
plot(d(1:N),':k')
hold on
% noisy process
plot(x(1:N),'k')
legend('d(n)', 'x(n)')
title('Noise-corrupted observation');
xlabel('samples n')
ylabel('amplitude')
axis([0 1000 -3 3])
grid on
end

%% Reference signal
% Here, the reference signal v2(n) is not known.
% In that case, it is possible to derive a reference signal by
% delaying the noisy process x(n) = d(n) + v1(n).
% The delayed signal x(n-n0) is used as the reference signal for % the canceller.
n0 = 25; % delay of 25 samples
len = N - n0; % reduced vector length
x_del = zeros(N,1); % create array of all zeros
% generate delayed signal
for i = 1:len
x_del(i) = x(i+n0);
end

if (add_figures) 
%plot of x_del(n)
figure(2)
subplot(2,1,1)
plot(x(1:N),':k')
%plot(x(1:N),':k')
hold on
plot(x_del(1:N),'k')
legend('x(n)', 'x(n-n0)')
title('Reference signal x(n-n0)');
xlabel('samples n')
ylabel('amplitude')
axis([0 1000 -3 3])
%axis([0 N -3 3])
grid on
end


% create arrays of all zeros
W_LMS = zeros(nord,1);
W_NLMS = zeros(nord,1);
W_KWONG = zeros(nord,1);
W_NEW = zeros(nord,1);
mu_kwong = 0.01; % initial value for each new run
mu_new = 0.01; % initial value for each new run
g_new = 0; % initial value for each new run
U = zeros(nord,1);
%P = ((1/delta)*eye(nord,nord));


for i=1:N
    U = [x_del(i);U(1:(nord-1))];
    x_n = x(i);

    %% NLMS Algorithm
    % Step 1: Filtering
    y_NLMS = (W_NLMS'*U);
    dhat_NLMS(i) = (dhat_NLMS(i)+y_NLMS);
    % Step 2: Error Estimation
    E_NLMS = (x_n-y_NLMS);
    err_NLMS(i) = err_NLMS(i)+E_NLMS;
    % Step 3: Tap-weight vector adaptation
    %W_NLMS = (W_NLMS+((beta/((norm(U)^2)))*conj(E_NLMS)*U));
    W_NLMS = (W_NLMS+ 2*mu_nlms*conj(E_NLMS)*U);
    % Step 4: Error performance
    MSE_NLMS(i) = norm(MSE_NLMS(i)+(abs(E_NLMS)^2));
    
    %% KWONG-NLMS Algorithm
    % Step 1: Filtering
    y_KWONG = (W_KWONG'*U);
    dhat_KWONG(i) = (dhat_KWONG(i)+y_KWONG);  
    % Step 2: Error Estimation
    E_KWONG = (x_n-y_KWONG);
    err_KWONG(i) = err_KWONG(i)+E_KWONG;
    % Step 3: Tap-weight vector adaptation
    mu_kwong = alpha_kwong*mu_kwong + gamma_kwong*(E_KWONG)^2;
    W_KWONG = W_KWONG+ (((2*mu_kwong)/(gamma_kwong+(U'*U)))*conj(E_KWONG)*U);
    % Step 4: Error performance
    MSE_KWONG(i) = norm(MSE_KWONG(i)+(abs(E_KWONG)^2));    

    
    %% NLMS-Proposed Algorithm
    % Step 1: Filtering
    y_NEW = (W_NEW'*U);
    dhat_NEW(i) = (dhat_NEW(i)+y_NEW);  
    % Step 2: Error Estimation
    E_NEW = (x_n-y_NEW);
    err_NEW(i) = err_NEW(i)+E_NEW;
    % Step 3: Tap-weight vector adaptation
    g_new = (beta_new*g_new) + (1-beta_new)*((conj(E_NEW)*U)/(gamma_new+(alpha_new*U'*U)));        
    mu_new = p_new*(norm(g_new)^2);
    W_NEW = W_NEW+ (mu_new*conj(E_NEW)*U)/(gamma_new+(mu_new*(U'*U)));
    % Step 4: Error performance
    MSE_NEW(i) = norm(MSE_NEW(i)+(abs(E_NEW)^2));  

end
end

%% Error performance
%MSE_LMS = MSE_LMS/R;
MSE_NLMS = MSE_NLMS/R;
MSE_KWONG = MSE_KWONG/R;
MSE_NEW = MSE_NEW/R;

% plot estimate of d(n)
% figure(3)
% plot(dhat_LMS(1:1000),'b');
% title('LMS - Estimate of d(n)');
% xlabel('samples n')
% ylabel('amplitude')
% axis([0 1000 -3 3])
% grid on
% figure(4)
% plot(dhat_NLMS(1:N),'b');
% title('NLMS - Estimate of d(n)');
% xlabel('samples n ')
% ylabel('amplitude')
% axis([0 N -3 3])
% grid on


%% Plot learning curves
% figure(10)
% plot(MSE_LMS(1:1000),':k')
% ylabel('ensemble-average squared error')
% xlabel('number of iterations')
% title('LMS - Convergence rate ')
% axis([0 1000 0 1])
% grid on

figure(11)
plot(MSE_NLMS(1:N),':k')
ylabel('ensemble-average squared error')
xlabel('number of iterations')
title('NLMS- Convergence rate ')
axis([0 N 0 1])
grid on
hold

%figure(12)
plot(MSE_KWONG(1:N),'k')
ylabel('ensemble-average squared error')
xlabel('number of iterations')
title('NLMS-KWONG Convergence rate ')
axis([0 N 0 1])
grid on


%figure(13)
plot(MSE_NEW(1:N),'b')
ylabel('ensemble-average squared error')
xlabel('number of iterations')
title('NLMS-Proposed Convergence rate ')
axis([0 N 0 1])
grid on


