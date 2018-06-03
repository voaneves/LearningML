#!/usr/bin/env python
# Title         : EN_BostonHousing.py
# Description   : Elastic Net is a linear regression model with L1 and L2 regu-
#                 larization. It's probably going to be useful as a baseline mo-
#                 del. Stacking is the next step and in this we'll find RMSE and
#                 plot the comparative graph of Real Prices vs Predicted Prices
# Author        : Neves4
# Outputs       : Figure with one plot      : 'Real Prices vs Predicted prices'
#                 Values                    : RMSE: 4.8273
#                                             R^2 score: 0.6268
#                                             CV Scores: 0.6495 (+/- 0.5872)
# License       : MIT License
#==============================================================================

##### IMPORTING #####
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn import datasets, model_selection
from sklearn.metrics import mean_squared_error, r2_score

sns.set() # set seaborn style

##### FUNCTIONS #####
def best_GridSearch(param_grid, X_train, Y_train, X_test, Y_test):
    """
    Function to optimize the fit with data. Recommended starting values:
        1. alpha = 1
        2. l1_ratio = 0.5 : It controls the balance between L1 and L2 penalties.
        3. tol = 1e-3 : This control the tolerance of aproximation.

    Best order for parameters tuning:
        1. Tune alpha
        2. Tune l1_ratio
        3. Tune tol
    """
    estimator = ElasticNet(alpha = 5.2,
                           #l1_ratio = 0.05,
                           tol = 1e-3,
                           random_state = 42
                           )

    regressor = model_selection.GridSearchCV(estimator=estimator, cv=5,
                                             param_grid=param_grid, verbose = 2)
    regressor.fit(X_train, Y_train)

    Y_pred = regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    cv_test = model_selection.cross_val_score(regressor, X_test, Y_test, cv=5)
    bestmodel = regressor.best_estimator_

    # OK great, so we got back the best estimator parameters as follows:
    print ("-----------  Best Estimator Parameters  -----------")
    print (regressor.best_params_)
    print ("-----------  ACCURACY ASSESSMENT -----------")
    print("RMSE: {:.4f}" .format(rmse))
    print("CV Scores - Test: {:.4f} (+/- {:.4f})" .format(cv_test.mean(),\
                                                   cv_test.std() * 2))

    return bestmodel

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
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,\
                                   test_size = 0.1, random_state = 42)

# ##### OPTIMIZATION OF THE MODEL #####
# param_grid = {'l1_ratio': [0.5]
#               # between 0,1 : [i/10.0 for i in range(6,10)]
#               # greater than 1 : range(2,10,2)
#              }
# best_est = best_GridSearch(param_grid, X_train, Y_train, X_test, Y_test)

# Fit dos dados com a ElasticNet
params = {'alpha': 5.2,
          'l1_ratio': 0.5,
          'random_state': 42}

en = ElasticNet(**params)

print("------- FITTING ElasticNet -------")
en.fit(X_train, Y_train)

Y_pred = en.predict(X_test)

##### ERROR #####
# Encontra o MSE, que será o benchmark para este algoritmo, para identificar
# quão boa foi sua aproximação
r2_score = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
cv_scores = model_selection.cross_val_score(en, X_test, Y_test, cv=5)

print("------- ACCURACY ASSESSMENT -------")
print("RMSE: {:.4f}" .format(rmse))
print("R^2 score: {:.4f}" .format(r2_score))
print("CV Scores: {:.4f} (+/- {:.4f})" .format(cv_scores.mean(),\
                                               cv_scores.std() * 2))

##### PLOTS #####
# Plot outputs using scatter. Ticks are diabled and everything else is the clea-
# nest that I could. Predicted prices vs Real Prices
custom_style = {'axes.labelcolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white'}
data = pd.DataFrame(data = {'Y_pred': Y_pred, 'Y_test': Y_test})
ax = sns.lmplot(x='Y_test', y='Y_pred', data = data, truncate=True, size=5)
ax.set_axis_labels("Real prices", "Predicted prices")
plt.tick_params(axis='both', colors='gray')
plt.title("Real vs Predicted prices on Boston Housing", fontweight = 'bold')
plt.tight_layout()
plt.show()
