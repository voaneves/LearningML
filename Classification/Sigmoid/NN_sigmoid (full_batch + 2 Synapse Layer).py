#!/usr/bin/env python
# Title         :ML_sigmoid (full_batch + 2 Synapse Layer).py
# Description   :This will transform to output with three layers (input, output
#                and a hidden layer), which forms a neural network. The input is
#                full-batched.
# Author        :iamtrask, modified by neves4
# License       : MIT License
#==============================================================================

# Importar numpy como dependência para as funções/matrizes nessa ML
import numpy as np

# Declarando função ordinal, para quando for printar as iterações
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])

# Criar função não-linear para avaliar a probabilidade de que aconteça evento.
# A função definida será a de sigmoid, com 1/(1+exp(-x))
def nonlin(x, deriv = False):
    if(deriv == True):
            return x*(1-x)
    return 1/(1+np.exp(-x))

# Dados para que aconteça o aprendizado de máquina, para seguir os exemplos. X
# é a entrada e Y a saída. O treino desta ML será feito por full-batch, porém
# de acordo com artigos no Stack Overflow é mais vantajoso o uso de mini-batches
X = np.array([  [0, 0, 1],     # ENTRADA
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]   ])

Y = np.array([  [0],           # SAÍDA
                [1],
                [1],
                [0] ])

# Como boa prática, para repetitividade de resultados, é necessário sempre ali-
# mentar os números aleatórios (random) do sistema
np.random.seed(1)

# Inicializar os neurônios da ML, de forma aleatória e com média 0 (esta função
# segue o princípio de função específica normal). Usar função de impressão
# 'print(syn0)' caso haja dúvida sobre como são Inicializados os neurônios
syn0 = 2*np.random.random((3, 4)) - 1 # 4 colunas para acamodar as 4 saídas
syn1 = 2*np.random.random((4, 1)) - 1 # 1 coluna pois será no final 4 x 1

# Os neurônios serão treinados t_ammount de vezes
t_ammount = 10000

for t in range(t_ammount):
    # L0 será sempre o input, L1 será o hidden layer, unido a L0 por syn0. A es-
    # timativa L2 do resultado é ligada a L1 por syn1
    L0 = X
    L1 = nonlin(np.dot(L0, syn0))
    L2 = nonlin(np.dot(L1, syn1))

    # Imprimir a primeira iteração, para avaliar mudanças no códiog. Está iden-
    # tado para melhor visualização
    if(t == 1):
        print(ordinal(t), "iteration:")
        print("\t" + str(L2).replace('\n','\n\t'))

    # Feita a estimativa, calcular o erro e quanto os pesos devem ser atualiza-
    # dos para a próxima estimativa. O quanto deve ser alterado no peso dos neu-
    # rônios é o erro vezes o gradiente da função
    L2_error = Y - L2
    L2_delta = L2_error * nonlin(L2, True)

    # Os pesos da syn0 também precisam ser atualizados, para encontrar os melho-
    # res valores
    L1_error = np.dot(L2_delta, syn1.T) # Resultado deve ser 4 x 4
    L1_delta = L1_error * nonlin(L1, True)

    # Atualização dos neurônios,
    syn1 += np.dot(L1.T, L2_delta)
    syn0 += np.dot(L0.T, L1_delta)

# Finalizada o treino dos pesos, temos a estimativa de output como L1
if(t == 9999):
    print(ordinal(t), "iteration:")
    print("\t" + str(L2).replace('\n','\n\t'))
