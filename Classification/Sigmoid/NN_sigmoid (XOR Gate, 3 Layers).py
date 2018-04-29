#!/usr/bin/env python
# Title         :ML_sigmoid (reescrita do zero, 3 layers).py
# Description   :This will transform to output with three layers (input, output
#                and a hidden layer), which forms a neural network. The input is
#                full-batched.
# Author        :neves4
# License       : MIT License
#==============================================================================

# Importar numpy como dependência para as funções/matrizes nessa ML
import numpy as np

# Math related functions - using sigmoid to this network
def ordinal(n):
    """If n is a cardinal, ordinal function will return it's ordinality"""
    return "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
def sigmoid(n):
    """This function returns the sigmoid of a given number/matrix"""
    return (1/(1+np.exp(-n)))
def deriv_sigmoid(n):
    """This function returns the slope of a sigmoid in a given number/matrix"""
    return (n*(1-n))

# Network related functions
def make_network(inputSize, hiddenSize, outputSize):
    """Given the synapses parameters, this function returns the model"""
    model = dict(
        syn0 = np.random.randn(inputSize, hiddenSize),
        syn1 = np.random.randn(hiddenSize, outputSize)
    )

    return model
def forwardprop(X, model):
    """Forward propagation (from input to output) update of the errors"""
    L1 = sigmoid(X @ model['syn0'])
    L2 = sigmoid(L1 @ model['syn1'])

    return L1, L2
def backprop(L1, L2, Y, model):
    """Backprop (output to input) to dicover how much to update the net"""
    # O algoritmo usa backpropagation, começando a calcular o erro pelos laers
    # mais externos, L2, calculando seu delta que irá para L1
    L2_error = Y - L2
    dL2 = L2_error * deriv_sigmoid(L2)
    error = np.mean(np.abs(L2_error))

    # Usando o L2_delta, o erro será propagado para L1
    L1_error = dL2 @ model['syn1'].T
    dL1 = L1_error * deriv_sigmoid(L1)

    return dL1, dL2, error
def sgd(X, Y, model):
    """Stochastic Gradient Descent, algorithm used to update the synapses"""
    # Atualização dos layers com forward propagation e depois
    L1, L2 = forwardprop(X, model)
    dL1, dL2, error = backprop(L1, L2, Y, model)

    model['syn1'] += alpha*(L1.T @ dL2) # Atualização das synapses com fator
    model['syn0'] += alpha*(X.T @ dL1) # 'alpha' maiores que a variação

    return model, error, L2

# Parametrização da rede neural e pontos de parada no treinamento
inputSize, hiddenSize, outputSize, max_epoch, alpha = [3, 4, 1, 10000, 10]

# Declaração de input (X), que também é o layer L0, e output esperado (Y)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
Y = np.array([[0, 1, 1, 0]]).T

# Como boa prática, é necessário realimentar o random para que os resultados se-
# jam homogêneos
np.random.seed(1)

# A estrutura principal da nossa rede neural será composta por três layers L e o
# número de sinapses deve ser igual a (n. de layers - 1) = 2
model = make_network(inputSize, hiddenSize, outputSize)

# Loop para treinamento da rede neural. Será o método de Stochastic Gradient
# Descent para atualizar a rede neural. Será imprimido o erro na primeira época
for epoch in range(1, max_epoch + 1):
    model, error, L2 = sgd(X, Y, model)

    if (epoch == 1): # Impressão do erro inicial
        print(ordinal(epoch) + " epoch:" + "\n\t" + "the error is " +
              str(error))

print("The final epoch is: " + str(epoch) + ". With that, the final error is "
       + str(error) + ".\nThe final L2 estimation is: \n\t" + str(L2).replace(
       '\n', '\n\t'))
