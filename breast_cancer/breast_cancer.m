
%loading data
data = load('C:\Users\akshaybahadur21\Desktop\breast_cancer\cancer_data.csv');
y = data(:, 2);
X = data(:, [3 ,4]);


%Plotting data
plotData(X,y);

%Computing Cost and Gradient

[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);

[cost, grad] = costFunction(initial_theta, X, y);


%Finding optimal theta
options = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plotting the boundary

plotDecisionBoundary(theta, X, y);
hold on;
xlabel('Mean Radius')
ylabel('Mean Texture')
legend('Malignant', 'Benign')
hold off;

