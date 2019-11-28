%% EXPECTATION PROPAGATION
clc;
close all;
clear all;
%% Setting up Simulation Parameters
N = 128; % Number of features
M = 96; % length of feature vector
p = 0.1; % Feature selection probability
AU = ceil(p*N);
pc = ones(N,1); % considering simple zero mean, unit variance complex coefficient for each feature
phi = exp(1i*2*pi*randn(M,N)); % generating the set of features
phi = phi*diag(1./vecnorm(phi)); % normalizing the feature vectors

% Features with noise
uset = randperm(N,AU);
h = sqrt(pc/2).*(normrnd(0,1,N,1) + 1i*normrnd(0,1,N,1));
yp = phi(:,uset)*h(uset);
tp = 20; % transmit power in dBm
np = -110; %noise power in dBm for transmission bandwidth 1 MHz
snr = tp-np;
ypn = awgn(yp,snr);
varn = 10^(-snr/10);
    
    
%% Expectation Propagation algo
% Initialization
m2 = zeros(N,1);
v2 = diag(p*pc);
v = inv((1/varn)*phi'*phi + inv(v2));
m = v*((1/varn)*phi'*ypn + inv(v2)*m2);

% Compute ratio distribution
count = 1;
while(count<10)
    count = count + 1;
    for i=1:N
        % Ratio Distribution Computation
        v2c = ((v(i,i))^-1 - (v2(i,i))^-1)^-1;
        m2c = v2c*(v(i,i)^-1*m(i)- (v2(i,i))^-1*m2(i));
        % Moment Computation
        G0 = (1-p)*cmplx_pdf(0,m2c,v2c) + p*cmplx_pdf(0,m2c,pc(i) + v2c); %first moment
        G1 = p*cmplx_pdf(0,m2c,pc(i) + v2c)*(m2c*pc(i)/(pc(i) + v2c)); %second moment 
        G2 = p*cmplx_pdf(0,m2c,pc(i) + v2c)*((abs(m2c*pc(i)/(pc(i) + v2c)))^2 + pc(i)*v2c/(pc(i) + v2c)); 
        Eq = G1/G0;
        Vq = G2/G0 - abs(Eq)^2;
        % Moment Matching: Update parameters for next iteration
        v2(i,i) = (1/Vq - 1/v2c)^-1;
        m2(i) = v2(i,i)*(Eq/Vq - m2c/v2c);
    end
    v = inv((1/varn)*phi'*phi + inv(v2));
    m = v*((1/varn)*phi'*ypn + inv(v2)*m2);
end



% threshold
threshold = zeros(N,1);
for i=1:N
    threshold(i) = log10(1 + pc(i)/v(i,i))/(1/v(i,i) - 1/(pc(i)+v(i,i)));
end

AU = find(abs(m).^2>=threshold);
[AU sort(uset')]