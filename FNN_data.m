load('Lab1_X.mat')
load('Lab1_y.mat')

% Create a feedforward neural network with one hidden layer of 10 neurons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Parameter tuning and training setting%%%%%
training_algorithms = {'trainlm', 'traingdm', 'trainrp'};
activations_funs = {'logsig', 'tansig', 'relu'};

% Set number of hidden layers and neurons for each layer
hiddenLayerSize = [5, 5, 5];
net = fitnet(hiddenLayerSize);

% Set activation function setting for each layer
for i = 1:length(hiddenLayerSize)
    net.layers{i}.transferFcn = activations_funs{1};
end

% Set training algorithms
net.trainFcn = training_algorithms{1}; % Levenberg-Marquardt backpropagation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Performance function
net.performFcn = 'mse'; % Mean Squared Error

% Divide data for training and testing
net.divideParam.trainRatio = 70 / 100;
net.divideParam.testRatio = 30 / 100;

% Train the neural network model
[net, tr] = train(net, X', y);

% Get the test indices
testInd = tr.testInd;

% Get the test data
X_test = X(testInd, :);
Y_test = y(testInd);

% Prediction
Y_pred = net(X_test');

% Assessment
mse_train = mean((y(tr.trainInd) - net(X(tr.trainInd, :)')).^2);
mse_test = mean((Y_test - Y_pred).^2);

% Display the results
fprintf('Train MSE: %.4f\n', mse_train);
fprintf('Test MSE: %.4f\n', mse_test);

figure
plot(Y_pred)
hold on
plot(Y_test)
legend('measured', 'predicted')

