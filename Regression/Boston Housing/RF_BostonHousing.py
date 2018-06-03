#!/usr/bin/env python
# Title         : RF_BostonHousing.py
# Description   : RandomForestRegressor is very similar to XGBRegressor. It fits
#                 a number of classifying decision trees on various subsamples
#                 of the data, using the average to improve accuracy and control
#                 overfitting. The difference between them is that Random Forest
#                 uses fully grown trees, trying to reduce variance but can't
#                 change bias. We continue moving towards stacking and Random
#                 Forest is a great algorithm for the baseline models.
# Author        : Neves4
# Outputs       : Figure with one plot      : 'Real Prices vs Predicted prices'
#                 Values                    : RMSE: 3.0215
#                                             R^2 score: 0.8538
#                                             CV Scores: 0.7580 (+/- 0.1581)
# License       : MIT License
#==============================================================================

##### IMPORTING #####
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, model_selection
from sklearn.metrics import mean_squared_error, r2_score

sns.set() # set seaborn style

##### FUNCTIONS #####
def best_GridSearch(param_grid, X_train, Y_train, X_test, Y_test):
    """
    Function to optimize the fit with data. Recommended starting values:
        1. n_estimators = 50 : First one to test, generally 0.5%-1% of number of
                               samples
        2. max_features = 'auto' : This algorithm can assess max_features by it-
                                   self, but it's nice to try another values
        3. max_depth = 5 : Starting with small depth
        4. min_samples_split = 2
        5. min_samples_leaf = 1
        6. bootstrap = True : Change if samples are drawn with replacements or
                              always the same as the original input

    Best order for parameters tuning:
        1. Tune n_estimators
        2. Tune max_features and max_depth
        3. Tune min_samples_split and min_samples_leaf
        4. Tune bootstrap
        6. Tune n_estimators again
    """
    estimator = RandomForestRegressor(n_estimators = 95,
                                      max_features = 'auto',
                                      max_depth = 18,
                                      min_samples_split = 2,
                                      min_samples_leaf = 1,
                                      bootstrap = True,
                                      random_state = 42)

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
# param_grid = {'n_estimators': [93, 94, 95, 96, 97],
#               #'colsample_bytree': [i/100.0 for i in range(78,87)]
#               # between 0,1 : [i/10.0 for i in range(6,10)]
#               # greater than 1 : range(2,10,2)
#              }
# best_est = best_GridSearch(param_grid, X_train, Y_train, X_test, Y_test)

# Fit dos dados com parâmetros encontrados na GridSearchCV na Random Forest
params = {'n_estimators': 95,
          'max_features': 'auto',
          'max_depth': 18,
          'min_samples_split': 2,
          'min_samples_leaf': 1,
          'bootstrap': True,
          'random_state': 42}

rf = RandomForestRegressor(**params)

print("------- FITTING RandomForestRegressor -------")
print("  DONE!")
rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

##### ERROR #####
# Encontra o MSE, que será o benchmark para este algoritmo, para identificar
# quão boa foi sua aproximação
r2_score = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
cv_scores = model_selection.cross_val_score(rf, X_test, Y_test, cv=5)

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
plt.title("Real vs Predicted prices on Boston Housing", fontweight='bold')
plt.tight_layout()
plt.show()
