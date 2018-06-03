#!/usr/bin/env python
# Title         : XGB_BostonHousing.py
# Description   : After using LinearRegression and GradientBoostingRegressor, we
#                 can further improve the predicitions with state-of-the-art
#                 algorithms, like XGBReegressor. It can use regularization and
#                 better predict correlations on this dataset. We plot RMSE per
#                 number of Boosters and we also plot the comparative graph of
#                 Real Prices vs Predicted Prices, with all features importances
# Author        : Neves4
# Outputs       : Figure with one plot      : 'XGBoost RMSE'
#                 Figure with two plots     : 'Predicted prices vs Real Prices'
#                                             'Importância das variáveis'
#                 Values                    : RMSE: 2.0278
#                                             R^2 score: 0.9341
#                                             CV Scores: 0.7756 (+/- 0.2291)
# License       : MIT License
#==============================================================================

##### IMPORTING #####
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn import datasets, model_selection
from sklearn.metrics import mean_squared_error, r2_score

plt.style.use('ggplot') # Customizando o estilo do matplotlib

##### FUNCTIONS #####
def plot_FeatureImportances(model, X, Y_test, Y_pred):
    """
    Plot the Feature Importances of a given model and also the predictions vs
    the actual values. This funcion outputs two graphs on the same figure
    """
    feature_importances = np.array(model.feature_importances_)
    feature_importances = 100*(feature_importances/feature_importances.max())
    pos = np.arange(feature_importances.shape[0]) + .5
    labels_X = np.array(X.columns.values)

    idx = np.argsort(feature_importances)

    feature_importances = np.array(feature_importances)[idx]
    labels_X = np.array(labels_X)[idx]

    plt.figure(figsize=(13, 6))

    # 1st graph - plot feature importances and their absolute value
    plt.subplot(1, 2, 1)
    plt.title('Importância das variáveis')
    plt.barh(pos, feature_importances, align='center')
    plt.yticks(pos, labels_X)
    plt.xlabel('Importância relativa')
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        length=0   )       # labels along the bottom edge are off

    # 2nd graph - scatter graph that compare estimated vs real prices
    plt.subplot(1, 2, 2)
    plt.scatter(Y_test, Y_pred, alpha=0.75, label='Índices comparados')
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        length=0   )       # labels along the bottom edge are off
    legend = plt.legend(loc='upper left', frameon=True, handletextpad=0.1)
    legend.get_frame().set_facecolor('white')
    plt.xlabel("Índice Real")
    plt.ylabel("Índice Estimado")
    plt.title("Comparativo entre índices reais e estimados")
    plt.tight_layout()
    plt.show()

def plot_PerformanceMetrics(model, error_used):
    """
    Assess performance metrics from given XGBoost model. It should be evaluated
    using RMSE during fit. Example of a model's fit funtion:

        eval_set = [(X_train, Y_train), (X_test, Y_test)]
        model.fit(X_train, Y_train, early_stopping_rounds = 100,
                  eval_metric = "rmse", eval_set = eval_set, verbose = True)
    """
    results = model.evals_result()
    epochs = len(results['validation_0'][error_used])
    x_axis = range(0, epochs)
    len_space = len(error_used) + 1
    title = "".join({'XGBoost', error_used.rjust(len_space).upper()})

    # Plot RMSE vs Iterations
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0'][error_used], label='Train')
    ax.plot(x_axis, results['validation_1'][error_used], label='Test')
    legend = ax.legend(loc='upper right', frameon=True)
    legend.get_frame().set_facecolor('white')
    ax.tick_params(axis = 'both',       # changes apply to the x-axis
                   which = 'both',      # major and minor ticks
                   length = 0)          # labels along the bottom edge
    plt.ylabel(error_used.upper())
    plt.xlabel('Number of estimators')
    plt.title(title)
    plt.show()

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
    estimator = xgb.XGBRegressor(n_estimators =  157,
                                 learning_rate = 0.1,
                                 max_depth = 5,
                                 min_child_weight = 2,
                                 gamma = 0.17,
                                 subsample = 0.84,
                                 colsample_bytree = 0.85,
                                 reg_alpha = 0.008,
                                 reg_lamba = 1.200
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
# param_grid = {'reg_lambda': [i/100.0 for i in range(115,125)]#,
#               #'colsample_bytree': [i/100.0 for i in range(78,87)]
#               # between 0,1 : [i/10.0 for i in range(6,10)]
#               # greater than 1 : range(2,10,2)
#              }
# best_est = best_GridSearch(param_grid, X_train, Y_train, X_test, Y_test)

# O método para fit dos dados do Boston serão os de Linear Regression
eval_set = [(X_train, Y_train), (X_test, Y_test)]
params = {'learning_rate': 0.0503,
          'n_estimators': 5000,
          'max_depth': 5,
          'min_child_weight': 2,
          'gamma': 0.17,
          'subsample': 0.84,
          'colsample_bytree': 0.85,
          'reg_alpha': 0.008,
          'reg_lambda': 1.200,
          'scale_pos_weight': 1,
          'seed': 42}

xgb1 = xgb.XGBRegressor(**params)

print("------- FITTING XGBOOST -------")
xgb1.fit(X_train, Y_train, early_stopping_rounds = 100, eval_metric = "rmse",
         eval_set = eval_set, verbose = 100)

Y_pred = xgb1.predict(X_test)

##### ERROR #####
# Encontra o MSE, que será o benchmark para este algoritmo, para identificar
# quão boa foi sua aproximação
r2_score = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
cv_scores = model_selection.cross_val_score(xgb1, X_test, Y_test, cv=5)

print("------- ACCURACY ASSESSMENT -------")
print("RMSE: {:.4f}" .format(rmse))
print("R^2 score: {:.4f}" .format(r2_score))
print("CV Scores: {:.4f} (+/- {:.4f})" .format(cv_scores.mean(),\
                                               cv_scores.std() * 2))

##### PLOTS #####
# Plot outputs using scatter. Ticks are diabled and everything else is the clea-
# nest that I could. The 1st graph - Featura Importances normalized with the
# highest value. Useful function here is print ("Feature Importances")
plot_PerformanceMetrics(xgb1, 'rmse')
plot_FeatureImportances(xgb1, X, Y_test, Y_pred)
