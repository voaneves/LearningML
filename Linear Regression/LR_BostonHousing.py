#!/usr/bin/env python
# Title         : Boston_LR.py
# Description   : With Linear Regression, we hope to accurately predict the Bos-
#                ton Housing prices using ScikitLearn techniques and datasets.
#                The algorithm used is the Ordinary Least Squares and in the
#                output, we're going to calculate the mean_squared_error and
#                plot a comparative graph of Real Prices vs Predicted Prices 
# Author        : Neves4
# License       : MIT License
#==============================================================================

##### IMPORTING #####
# Importar numpy como dependência para as funções/matrizes nessa ML. Para plotar
# usaremos o pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar também os datasets (pois utilizaremos o load_boston) e serão usadas
# as funções shuffe (para embaralhar os dados) e mean_square_error (para calcu-
# lar o erro do algoritmo em questão)
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

##### DECLARING AND TRAINING #####
# Carregamento do dataset do boston, conversão para o framework pandas e como a
# nomenclatura não é automática, foi dado valor às colunas da tabela do pandas.
# Para verificar como estão os dados, chamar print(boston_pd.head())
boston = datasets.load_boston()
boston_pd = pd.DataFrame(boston.data)
boston_pd.columns = boston.feature_names

# É necessária então a divisão dos datasets, pelo método train_test_split. Para
# encontrar o tamanho de cada tensor que foi dividido, print(X_train.shape)
X, Y = boston_pd, boston.target
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, \
                                   test_size = 0.1, random_state = 42)

# O método para fit dos dados do Boston serão os de Linear Regression
lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

##### ERROR #####
# Encontra o MSE, que será o benchmark para este algoritmo, para identificar
# quão boa foi sua aproximação
mse = mean_squared_error(Y_test, Y_pred)
print("%.4f" %mse)

##### PLOT #####
# These are the "Tableau 20" colors as RGB. Using number 4
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib
# accepts
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

# Plot outputs, using scatter and tableau n. 4. Ticks are diabled and everything
# else is the cleanest that I could
plt.style.use('ggplot')
plt.scatter(Y_test, Y_pred, c=tableau20[4], alpha=0.75, label=\
            'Comparative Prices')
plt.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    length=0   )       # labels along the bottom edge are off
plt.legend(loc='upper left', frameon=True)
plt.xlabel("Real prices")
plt.ylabel("Predicted prices")
plt.title("Predicted prices vs Real Prices")
plt.show()
