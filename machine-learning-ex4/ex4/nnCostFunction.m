function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%ģ���������ǰ�򴫲����̣�Theta1��Theta2���Ѿ�ѵ���õ�
a1 = X;           %��һ��

a1 = [ones(m,1),a1];
z2 = a1 * Theta1';
a2 = sigmoid(z2);       %�ڶ���

a2 = [ones(m,1),a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);       %�����㣬a3�ɿ�����������h(x)

for k = 1:num_labels
    %���ݹ�ʽ�����㵱kΪ��ͬ����ʱ�����֮�ͣ�ͨ��a3(:,k)��ȡ����Ӧ������(y==k)�����������
    J = J + (1/m) * (-(y == k)' * log(a3(:,k)) - (1-(y == k))' * log(1 - a3(:,k)));
end

%���浥���������򻯵�β��
sum = 0;
for i = 1:hidden_layer_size
    for j = 2:input_layer_size + 1        % Theta1��һ�в�����
        sum = sum + Theta1(i,j)^2;
    end
end

for i = 1:num_labels
    for j = 2:hidden_layer_size + 1      %Theta2��һ�в�����
        sum = sum + Theta2(i,j)^2;
    end
end

J = J + lambda/(2*m) * sum;       %�������򻯵���ʧ����J

% -------------------------------------------------------------

%���򴫲��㷨
delta_1 = 0;
delta_2 = 0;
for t = 1:m          %����Ҫ�󣬷ֳ�m��ѭ��ʵ�ַ��򴫲���m����ѵ�������ĸ��� 
    d_3 = a3(t,:)' - ([1:num_labels] == y(t,1))';         %��Ӧstep2��d_3Ϊ10x1����
    d_2 = Theta2(:,2:end)' * d_3 .* sigmoidGradient(z2(t,:)'); %��Ӧstep3��ע��Theta2�ĵ�һ��Ϊbias������ȥ��d_2Ϊ25x1����
    delta_1 = delta_1 + d_2 * a1(t,:);   %��Ӧstep4
    delta_2 = delta_2 + d_3 * a2(t,:);   %��Ӧstep4
end

%�������򻯵��ݶ�
Theta1(:,1) = 0;        %�ѵ�һ��ȫ��0������Ҫ��������
Theta1_grad = delta_1 / m + lambda / m * Theta1;
Theta2(:,1) = 0;
Theta2_grad = delta_2 / m + lambda / m * Theta2;

% ========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
