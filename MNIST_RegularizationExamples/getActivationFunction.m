function func = getActivationFunction(number)

%Select between the following activation functions
% % https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
% 1. Sigmoid
% 2. Hyperbolic Tangent
% 3. Rectified Linear Unit
% 4. Leaky ReLU
% 5. Exponential
% 6. Exponential linear unit

switch number
    case 1
        func = @(inMatrix) internal_sigmoid(inMatrix);
    case 2
        func = @(inMatrix) tanh(inMatrix);
    case 3
        func = @(inMatrix) internal_relu(inMatrix);
    case 4
        func = @(inMatrix) internal_leakyrelu(inMatrix);
    case 5
        func = @(inMatrix) exp(inMatrix);
    case 6
        disp(['Selected the value of alpha in exponential linear unit as 0.2,' ...
            newline ' change in getActivation function if you intend to change this parameter']);
        func = @(inMatrix) max(0, inMatrix) + 0.2*exp(min(0, X)) -1;
    otherwise
        disp('Defaulting to sigmoid activation function because nothing is selected.');
        func = @(inMatrix) internal_sigmoid(inMatrix);
end

end