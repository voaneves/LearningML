
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
                [0],
                [1],
                [1] ])

# Como boa prática, para repetitividade de resultados, é necessário sempre ali-
# mentar os números aleatórios (random) do sistema
np.random.seed(1)

# Inicializar os neurônios da ML, de forma aleatória e com média 0 (esta função
# segue o princípio de função específica normal)
syn0 = 2*np.random.random((3,1)) - 1
# Usar função de impressão print(syn0) caso haja dúvida sobre como são Iniciali
# zados os neurônios (synapses)

# Os neurônios serão treinados t_ammount de vezes
t_ammount = 10000

for t in range(t_ammount):

    # I0 será sempre o input, I1 será o output, unidos pelas synapses. A estima-
    # tiva I1 do resultado é o sigmoid da multiplicação de matrizes entre I0 e
    # syn0
    L0 = X
    L1 = nonlin(np.dot(L0, syn0))

    # Imprimir a primeira iteração, para avaliar mudanças no códiog. Está iden-
    # tado para melhor visualização
    if(t == 1):
        print(ordinal(t), "iteration:")
        print("\t" + str(L1).replace('\n','\n\t'))

    # Feita a estimativa, calcular o erro e quanto os pesos devem ser atualiza-
    # dos para a próxima estimativa. O quanto deve ser alterado no peso dos neu-
    # rônios é o erro vezes o gradiente da função
    L1_error = Y - L1
    L1_delta = L1_error * nonlin(L1, True)

    # Atualização dos neurônios,
    syn0 += np.dot(L0.T, L1_delta)

# Finalizada o treino dos pesos, temos a estimativa de output como L1
if(t == 9999):
    print(ordinal(t), "iteration:")
    print("\t" + str(L1).replace('\n','\n\t'))


# Baseado (quase idêntico) ao script em
# http://iamtrask.github.io/2015/07/12/basic-python-network/.
