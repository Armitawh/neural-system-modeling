% Read the CSV file into a DataFrame
filename = 'STR_data.csv';
STR_data = readtable(filename);

% Access data in the DataFrame using column names
t = STR_data.Time;
u = STR_data.Caf;
y = STR_data.Ca;

% Downsample the data
wave_duration = 100;
sample_rate = 100;
sample_decimated_rate = 10;
samples = sample_rate * wave_duration;
samples_decimated = sample_decimated_rate * wave_duration;

t = 0:1/sample_decimated_rate:wave_duration;
u = downsample(u, sample_rate/sample_decimated_rate);
y = downsample(y, sample_rate/sample_decimated_rate);

% Prepare the training data
t_train = t(1:floor(0.7 * samples_decimated));
u_train = con2seq(u(1:floor(0.7 * samples_decimated))');
y_train = con2seq(y(1:floor(0.7 * samples_decimated))');

% Create the model
net = timedelaynet(1:50,10.);
[X, Xi, Ai, Y] = preparets(net, u_train, y_train);

% Train the model
net = train(net, X, Y, Xi, Ai);

% Make predictions
y_hat_train = net(u_train, Xi, Ai);
y_hat_train = cell2mat(y_hat_train);
y_train = cell2mat(y_train);

% Prepare the test data
t_test = t(floor(0.7 * samples_decimated) + 1:samples_decimated);
u_test = con2seq(u(floor(0.7 * samples_decimated) + 1:samples_decimated)');
y_test = con2seq(y(floor(0.7 * samples_decimated) + 1:samples_decimated)');
[X, Xi, Ai, Y] = preparets(net, u_test, y_test);

% Make predictions
y_hat_test = net(u_test, Xi, Ai);
y_hat_test = cell2mat(y_hat_test);
y_test = cell2mat(y_test);

% Calculate MSE
mse_train = mean((y_hat_train - y_train).^2)
mse_test = mean((y_hat_test - y_test).^2)

% Plotting the results
figure; % Opens a new figure window
plot(t_train, y_train, t_train, y_hat_train, 'LineWidth', 1.5);
legend('System output', 'Model output');
xlabel('Time (s)');
ylabel('Output');

figure; % Opens a new figure window
plot(t_test, y_test, t_test, y_hat_test, 'LineWidth', 1.5);
legend('System output', 'Model output');
xlabel('Time (s)');
ylabel('Output');

