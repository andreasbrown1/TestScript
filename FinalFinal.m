format short

%A = [-1,-1,0;-1,0,-1;0,-1,1];
A = [1,1,0;1,1,0];
rankA = rank(A);
AminusM = norm_pseudo_inverse(A);
Aplus = general_pseudo_inverse(A);


inverseA = general_inverse(A);

t = [1;2;3;4;5;6;7;8];
x = [-2;-1;1;2;1;-2;1;1];
y = [0;1;-2;-1;2;1;-1;-2];
CorrXY(x, x, -1);
CorrXY(y, y, 3);
CorrXY(x, y, 2);


meanx = mean(x);
meany = mean(y);
CorrXY(x, x, -1)-(meanx^2);
CorrXY(x, y, 0)-(meanx*meany);

apriori = 1;
Qv = (1/6)*[1,2,-1;2,4,-2;-1,-2,1];
correlationmat = ccmat(Qv);
v = [0.7;-0.5;-0.2];
deltal = [0;12;0];
P = eye(3);
deltav = -(Qv*P)*deltal;
v = v +deltav;
%Global test
T = (v'*P*v)/(apriori^2);
%Local test
w = v./(apriori*sqrt(diag(Qv)));
correlationmat = ccmat(Qv);

S = [9.1388,0.7640;0.7640,3.3333];
W = [35.4339;14.1840];

theta = (inv(S))*W;

% n = 5;
% A = zeros(5, 2);
% A(:, 1) = [1;1;1;1;6];
% A(:, 2) = [7;-1;-5;-7;1];
% l = [15;-1;-9;-13;8+15];
% P = eye(n);
% N = A'*P*A;
% U = A'*P*l;
% x = (inv(N))*U;
% v = A*x-l;
% 
% deltal = [0;0;0;0;15];
% deltax = (inv(N))*(A'*P*deltal);
% 
% apriori = 1;
% 
% %Global test
% T = (v'*P*v)/(apriori^2);
% Q = inv(P);
% Qv = Q-(A*(inv(A'*P*A))*A');
% deltav = -(Qv*P)*deltal;
% 
% %Local test
% w = v./(apriori*sqrt(diag(Qv)));
% correlationmat = ccmat(Qv);


f = [6;5;14;20;29;32;30;27;18;11;7;4];
F = [4.5;6.4;12.1;19.4;26.5;31.1;31.1;26.5;19.4;12.1;6.4;4.5];

test = ((f-F).^2)./F;

Data = [1.0; 0.5; -0.5];
RandomWalk = zeros(length(Data), 11); %A matrix to store key information for each epoch
RandomWalk(:, 1) = 1:length(Data);

Rk = 0.25; %Measurement covariance matrix
Qk = 1; %Process noise (spectral density matrix)
Hk = 1; %Design matrix

%Initial, apriori starting values
xkhatplus = 0; %State vector
Pkplus = 0; %Parameter (state) covariance matrix

for i=1:length(Data) %For all epochs, apply the recursive Kalman filter
    
    phik = 1; %Transition matrix
    %deltat = 0.5; %Time increment between epochs

    xkhatminus = phik * xkhatplus; %Predict/propagate the state vector
    Pkminus = (phik*Pkplus*phik')+Qk; %Predict/propagate the parameter covariance matrix

    zk = Data(i); %Position observation at this time epoch
    
    Kk = (Pkminus*Hk')*(inv((Hk*Pkminus*Hk')+Rk)); %Compute the Kalman gain
    
    RandomWalk(i, 2) = xkhatminus;
    RandomWalk(i, 3) = Pkminus;
    RandomWalk(i, 4) = zk;
    RandomWalk(i, 5) = Kk;
    
    xkhatplus = xkhatminus + Kk*(zk-(Hk*xkhatminus)); %Update the state vector
    Pkplus = (1-(Kk*Hk))*Pkminus; %Update the parameter covariance matrix
    
    %Analyze innovation sequence and its covariance
    vk = zk - (Hk*xkhatminus); %Innovation sequence
    Cvk = (Hk*Pkminus*Hk')+Rk; %Innovation sequence covariance matrix
    
    %Store key results
    RandomWalk(i, 6) = xkhatplus;
    RandomWalk(i, 7) = Pkplus;
    RandomWalk(i, 8) = vk;
    RandomWalk(i, 9) = Cvk;
    
    %Global test
    T = vk'*(inv(Cvk))*vk; %Compute test statistic
    RandomWalk(i, 10) = T;
    if (T > 3.84) %Compare with threshold value from Chi-Square table
        RandomWalk(i, 11) = 1;
    end
    
end


%% Function Definitions

function AL = left_pseudo_inverse(A)
AL = (inv(A'*A))'*A';
end

function AR = right_pseudo_inverse(A)
AR = A'*(inv(A*A'));
end

function Aminus = general_inverse(A)
r = rank(A);
n = size(A, 1);
m = size(A, 2);
A11 = A(1:r, 1:r);
A11inv=inv(A11);
Aminus = zeros(n, m);
Aminus(1:r, 1:r) = A11inv;
end

function AminusM = norm_pseudo_inverse(A)
AminusM = A'*general_inverse(A*A');
end

function Aplus = general_pseudo_inverse(A)
Aplus = A'*general_inverse(A*A')*A*general_inverse(A'*A)*A';
end

function n = compute_norm(v) %Works for both vectors and matrices
n = sqrt(sum((v.^2),'all'));
end

function out = CorrXY(x, y, tau)
if tau >= 0
    xx = x;
    yy = y;
else
    xx = y;
    yy = x;
    tau = -tau;
end

num = length(xx);
yy = [yy(1+tau:num); zeros(tau, 1)];
out = xx'*yy/(num-tau);

end

function rho = ccmat(Qv)
n = length(Qv);
rho = zeros(n, n);
for i = 1:n
    for j = 1:n
        rho(i, j) = Qv(i, j)/(sqrt(Qv(i, i))*sqrt(Qv(j, j)));
    end
end
end

