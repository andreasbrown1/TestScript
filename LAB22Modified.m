%data=RemoteArray(:,4:5);
%M = data;

testdata = [4847.073, 146.266; 4992.748, 144.844; 5136.899, 143.688; 5280.325, 143.156; 5452.985, 142.032];
M = testdata;

A = M(1:end-1,1); %Previous phase
B = M(2:end  ,1); %Current phase
C = M(1:end-1,2); %Previous rate
D = M(2:end  ,2); %Current rate
%Y = A+(D+C/2)*1; %I think you more need brackets here, replace with Y = A +((D+C)/2)*1;
Y = A +((D+C)/2)*1; %Prediction
%Z2 = Y-B; %Use absolute value here, replace with Z2 = abs(Y - B);
Z2 = abs(Y - B); %Absolute difference
plot(Z2) %Plot absolute difference
%yline(1); %Plot a horizontal line representing the threshold
hold on
plot([0, 3600], [1, 1]);