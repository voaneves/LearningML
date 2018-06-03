#!/usr/bin/env python
# Title         : SVR_BostonHousing.py
# Description   : Support Vector Machines are very widely used on regression mo-
#                 dels and there are some examples in which they're use as one
#                 of the 1st level models. As we're moving towards stacking, SVR
#                 is going to be useful on predicting our test data. We'll find
#                 RMSE and plot the comparative graph of Real Prices vs Predic-
#                 ted Prices
# Author        : Neves4
# Outputs       : Figure with one plot      : 'Real Prices vs Predicted prices'
#                 Values                    : RMSE: 2.0278
#                                             R^2 score: 0.9341
#                                             CV Scores: 0.7756 (+/- 0.2291)
# License       : MIT License
#==============================================================================

##### IMPORTING #####
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVR
from sklearn import datasets, model_selection
from sklearn.metrics import mean_squared_error, r2_score

sns.set() # set seaborn style

##### FUNCTIONS #####
def best_GridSearch(param_grid, X_train, Y_train, X_test, Y_test):
    """
    Function to optimize the fit with data. Recommended starting values:
        1. max_depth = 5 : This should be between 3-10. I’ve started with 5 but
        you can choose a different number as well. 4-6 can be good starting
        points.
        2. min_child_weight = 1 : A smaller value is chosen because it is a
        highly imbalanced class problem and leaf nodes can have smaller size
        groups.
        3. gamma = 0 : A smaller value like 0.1-0.2 can also be chosen for
        starting. This will anyways be tuned later.
        4. subsample, colsample_bytree = 0.8 : This is a commonly used used
        start value. Typical values range between 0.5-0.9.
        5. scale_pos_weight = 1: Because of high class imbalance.

    Best order for parameters tuning:
        1. Tune n_estimators with eta = 0.1
        2. Tune max_depth and min_child_weight
        3. Tune gamma
        4. Tune subsample and colsample_bytree
        5. Tune lambda and alpha
        6. Decrease learning_rate while increasing n_estimators proportionally
        (cv function)
    """
    estimator = SVR(kernel = 'linear',
                    #C = 1.0,
                    #epsilon = 0.05,
                    #gamma = 'auto'
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
# param_grid = {'C': [i/100.0 for i in range(10, 12)],
#               'gamma': [0.1, 0.001, 0.0001, 0.00001, 0.000001, 1],
#               'epsilon': [i/1000.0 for i in range(10, 12)]
#               #'colsample_bytree': [i/100.0 for i in range(78,87)]
#               # between 0,1 : [i/10.0 for i in range(6,10)]
#               # greater than 1 : range(2,10,2)
#              }
# best_est = best_GridSearch(param_grid, X_train, Y_train, X_test, Y_test)

# A partir dos parâmetros encontrados no GridSearchCV, fit com a SVR
params = {'kernel': 'linear',
          'C': 0.11,
          'epsilon': 0.011,
          'gamma': 0.1}

svr = SVR(**params)

print("------- FITTING SVR -------")
svr.fit(X_train, Y_train)

Y_pred = svr.predict(X_test)

##### ERROR #####
# Encontra o MSE, que será o benchmark para este algoritmo, para identificar
# quão boa foi sua aproximação
r2_score = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
cv_scores = model_selection.cross_val_score(svr, X_test, Y_test, cv=5)

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
