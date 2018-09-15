function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%模拟神经网络的前向传播过程，Theta1和Theta2是已经训练好的
a1 = X;     %第一层

a1 = [ones(m,1) a1];  
a2 = sigmoid(a1 * Theta1');       %第二层

a2 = [ones(m,1) a2];
a3 = sigmoid(a2 * Theta2');       %第三层

 %求a3矩阵每行的最大值row_max及其下标p,根据题目要求下标10代表的是数字0
[row_max,p] = max(a3,[],2);     

% =========================================================================

end
