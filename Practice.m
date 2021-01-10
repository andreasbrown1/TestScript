format short
Data = load('Lab0Data.txt');
%Data = Data(1:20, :);
k = Data(:, 1);
y = Data(:, 2);
%n = length(Data);
n = size(Data, 1);
u = 3;

%Regular parametric batch least squares
x0 = zeros(u, 1);
Cl = eye(n);
A = [k.^2, k, ones(n, 1)];
w = -y;
N = A'*(inv(Cl))*A;
U = A'*(inv(Cl))*w;
delta = -(inv(N))*U;
xhat = x0+delta;
vhat = A*delta+w;
aposteriori = (vhat'*(inv(Cl))*vhat)/(n-u);
Cxhat = inv(N);

%Summation of normals
%Now assume that the measurements are made in two separate, independent batches
%Ex: First batch contains 11 rows of data, second batch contains 10 rows of data
% n1 = 11;
% n2 = 10;
% Cl1 = Cl(1:n1,1:n1);
% Cl2 = Cl(n1+1:n1+n2,n1+1:n1+n2);
% A1 = A(1:n1,:);
% A2 = A(n1+1:n1+n2,:);
% w1 = w(1:n1,:);
% w2 = w(n1+1:n1+n2,:);
% N1 = A1'*(inv(Cl1))*A1;
% N2 = A2'*(inv(Cl2))*A2;
% U1 = A1'*(inv(Cl1))*w1;
% U2 = A2'*(inv(Cl2))*w2;
% deltaSON = -(inv(N1+N2))*(U1+U2);
% xhatSON = x0+deltaSON;

%((i-1)*ni+1):(i*ni)
%Note: There might be cases when we are given groups of data and we have to populate the matrices in real time instead of just partioning the original large matrices.

% numGroups = n;
% Nmatrices = cell(numGroups, 1);
% Umatrices = cell(numGroups, 1);
% Nmatrices{2} = xhat;
% for i = 1:numGroups
%     ni = n/numGroups;
%     Cli = Cl(((i-1)*ni+1):(i*ni),((i-1)*ni+1):(i*ni));
%     Ai = A(((i-1)*ni+1):(i*ni),:);
%     wi = w(((i-1)*ni+1):(i*ni),:);
%     Ni = Ai'*(inv(Cli))*Ai;
%     Nmatrices{i} = Ni;
%     Ui = Ai'*(inv(Cli))*wi;
%     Umatrices{i} = Ui;
% end
% 
% sumNi = zeros(u, u);
% sumUi = zeros(u, 1);
% for i = 1:numGroups
%     sumNi = sumNi + Nmatrices{i};
%     sumUi = sumUi + Umatrices{i};
% end
% deltaSON = -(inv(sumNi))*sumUi;
% xhatSON = x0 + deltaSON;

%Maybe try separating the observations into groups instead of the matrices

%Sequential least squares
%Let the initial solution consist of the first three observations (at least three observations are needed to obtain a solution)
Data0 = Data(1:u,:);
x0 = zeros(u, 1);

Cl1 = eye(u);
A1 = [Data0(:,1).^2, Data0(:,1), ones(u, 1)];
w1 = -Data0(:,2);
N1 = A1'*(inv(Cl1))*A1; 
U1 = A1'*(inv(Cl1))*w1;
deltaminus = -(inv(N1))*U1;
Cdeltaminus = (inv(N1));
    
% for i = (u+1):n %4:21 
%     Cli = 1;
%     Ai = [Data(i, 1).^2, Data(i, 1), 1];
%     wi = -Data(i, 2);
%     Ni = Ai'*(inv(Cli))*Ai;
%     Ui = Ai'*(inv(Cli))*wi;
%     K = Cdeltaminus*Ai'*(inv(Cli+(Ai*Cdeltaminus*Ai')));
%     %K = (inv(N1))*Ai'*(inv(Cli+(Ai*(inv(N1))*Ai'))); %Doesn't work
%     deltaplus = deltaminus - (K*(Ai*deltaminus+wi));
%     Cdeltaplus = Cdeltaminus - (K*Ai*Cdeltaminus);
%     xhatSLS = x0 + deltaplus;
%     deltaminus = deltaplus;
%     Cdeltaminus = Cdeltaplus;
%     %N1 = Ni;
%     %Need to update N1?
% 
%     
% end

%Kalman filter (assume random constant)

R = 1; %Measurement covariance matrix
Q = 0; %Spectral density matrix (process noise)
%Initial, apriori starting values
xhatplus = deltaminus; %State vector (based on first 3 epochs)
Pplus = Cdeltaminus; %Parameter (state) covariance matrix (based on first 3 epochs)
% xhatplus = zeros(3, 1);
% Pplus = eye(3);

for i = (u+1):n %For all remaining epochs, apply the recursive Kalman filter
    
    phi = 1; %Transition matrix
    xhatminus = phi * xhatplus; %Predict/propagate the state vector
    Pminus = (phi*Pplus*phi')+Q; %Predict/propagate the parameter covariance matrix
    
    H = [Data(i, 1).^2, Data(i, 1), 1]; %Design matrix
    z = Data(i, 2); %Observation
    
    K = (Pminus*H')*(inv((H*Pminus*H')+R)); %Compute the Kalman gain
    xhatplus = xhatminus + K*(z-(H*xhatminus)); %Update the state vector
    Pplus = (eye(u)-(K*H))*Pminus; %Update the parameter covariance matrix
    
end

% x0 = zeros(u, 1);
% Cl = eye(n);
% A = [k.^2, k, ones(n, 1)];
% w = -y;
% N = A'*(inv(Cl))*A;
% U = A'*(inv(Cl))*w;
% delta = -(inv(N))*U;
% xhat = x0+delta;
%Prediction, measurement, update
