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
import os
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')

# Importar também os datasets (pois utilizaremos o load_boston) e serão usadas
# as funções shuffe (para embaralhar os dados) e mean_square_error (para calcu-
# lar o erro do algoritmo em questão)
from sklearn import model_selection
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

##### DECLARING AND TRAINING #####
# Carregamento do dataset do boston, conversão para o framework pandas e como a
# nomenclatura não é automática, foi dado valor às colunas da tabela do pandas.
# Para verificar como estão os dados, chamar print(boston_pd.head())
current_folder = os.path.join(os.path.dirname(__file__),\
                              'Input\gun-violence-data_01-2013_03-2018.csv')
df = pd.read_csv(current_folder, encoding="utf8")

sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.tight_layout()
plt.show()
