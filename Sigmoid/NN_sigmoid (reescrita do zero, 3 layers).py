#!/usr/bin/env python
# Title         :ML_sigmoid (reescrita do zero, 3 layers).py
# Description   :This will transform to output with three layers (input, output
#                and a hidden layer), which forms a neural network. The input is
#                full-batched.
# Author        :neves4
#==============================================================================

# Importar numpy como dependência para as funções/matrizes nessa ML
import numpy as np

# Declarando função ordinal, para quando for printar as iterações. A função não-
# linear utilizada será a de sigmoid, com a sua derivada (usada no Gradient Des-
# cent) deriv_sigmoid
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
sigmoid = lambda n: (1/(1+np.exp(-n)))
deriv_sigmoid = lambda n: (n*(1-n))

# Declaração de input (X) e output esperado (Y)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
Y = np.array([[0, 1, 1, 0]]).T

# Como boa prática, é necessário realimentar o random para que os resultados se-
# jam homogêneos
np.random.seed(1)

# A estrutura principal da nossa rede neural será composta por três layers L e o
# número de sinapses deve ser igual a (n. de layers - 1) = 2. O layer L0 será
# sempre o input, será adicionado para visualização dos layers
syn0 = np.random.randn(3, 4)
syn1 = np.random.randn(4, 1)
L0 = X

# Parametrização da rede neural e pontos de parada no treinamento
epoch, max_epoch = [0, 10000]

# Loop para treinamento da rede neural
for epoch in range(max_epoch):
    # Atualização dos layers de acordo com os pesos
    L1 = sigmoid(L0 @ syn0)
    L2 = sigmoid(L1 @ syn1)

    # O algoritmo usa backpropagation, começando a calcular o erro pelos laers
    # mais externos, L2, calculando seu delta que irá para L1
    L2_error = Y - L2
    L2_delta = L2_error * deriv_sigmoid(L2)

    if (epoch == 0):
        error = np.mean(np.abs(L2_error))
        print(ordinal(epoch + 1) + " epoch:" + "\n\t" + "the error is " +
              str(error))

    # Usando o L2_delta, o erro será propagado para L1
    L1_error = L2_delta @ syn1.T
    L1_delta = L1_error * deriv_sigmoid(L1)

    syn1 += 10*(L1.T @ L2_delta)
    syn0 += 10*(L0.T @ L1_delta)

error = np.mean(np.abs(L2_error))
print("The final epoch is: " + str(epoch+1) + ". With that, the final error is "
       + str(error) + ".\nThe final L2 estimation is: \n\t" + str(L2).replace(
       '\n', '\n\t'))
