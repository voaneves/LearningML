#!/usr/bin/env python
# Title         : Boston_LR.py
# Description   : With Linear Regression, we estabilished our baseline model.
#                 Moving to Gradient Boosting Regressor from sklearn, we expect
#                 to improve accuracy R^2 score and CV score, for better predic-
#                 tions. We also plot a comparative graph of Real Prices vs pre-
#                 dicted Prices, with all features' importances
# Author        : Neves4
# Outputs       : Figure with two plots: 'Predicted prices vs Real Prices'
#                                        'Importância das variáveis'
#                 Values: RMSE: 2.1041
#                         R^2 score: 0.9117
#                         CV Scores: 0.61 (+/- 0.52)
# License       : MIT License
#==============================================================================

##### IMPORTING #####
# Importar numpy como dependência para as funções/matrizes nessa ML. Para plotar
# usaremos o pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot') # Customizando o estilo do matplotlib

# Importar também os datasets (pois utilizaremos o load_boston) e serão usadas
# as funções shuffe (para embaralhar os dados) e mean_square_error (para calcu-
# lar o erro do algoritmo em questão)
from sklearn import datasets
from sklearn import model_selection
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score

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

def GradientBooster(param_grid, n_jobs):
    estimator = ensemble.GradientBoostingRegressor(n_estimators = 250,\
                                                   max_depth = 7,\
                                                   min_samples_split = 8,\
                                                   min_samples_leaf = 2,\
                                                   learning_rate = 0.1,\
                                                   subsample = 0.955,\
                                                   max_features =  'sqrt',\
                                                   random_state = 42)
    classifier = model_selection.GridSearchCV(estimator=estimator, cv=5,\
                                              param_grid=param_grid,\
                                              n_jobs=n_jobs)
    classifier.fit(X_train, Y_train)
    print ("Best Estimator learned through GridSearch")
    print (classifier.best_estimator_)
    return classifier.best_estimator_

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

# O método para fit dos dados do Boston serão os de Linear Regression
params = {'n_estimators': 250, 'max_depth': 7, 'min_samples_split': 8,\
         'min_samples_leaf': 2, 'learning_rate': 0.1, 'loss': 'ls',\
         'max_features': 'sqrt', 'subsample': 0.955, 'random_state': 42}
gbt = ensemble.GradientBoostingRegressor(**params)
gbt.fit(X_train, Y_train)

Y_pred = gbt.predict(X_test)

###################################################################
##### FOR FINDING THE OPTIMAL VALUES, USING GradientBooster() #####
# Guide for using this: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
# param_grid={'subsample': [0.95, 0.955, 0.96]#,
#             #'min_samples_leaf': [1, 2, 3, 4, 5, 6]
#             }
# n_jobs=1
# best_est=GradientBooster(param_grid, n_jobs)
#
# cv_scores = model_selection.cross_val_score(best_est, X, Y, cv=5)
# print("CV Scores: {:.2f} (+/- {:.2f})" .format(cv_scores.mean(),\
#                                                cv_scores.std() * 2))
#
# #OK great, so we got back the best estimator parameters as follows:
# print ("Best Estimator Parameters")
# print ("---------------------------")
# print ("n_estimators: %d" %best_est.n_estimators)
# print ("max_depth: %d" %best_est.max_depth)
# print ("Learning Rate: %.2f" %best_est.learning_rate)
# print ("min_samples_split: %d" %best_est.min_samples_split)
# print ("min_samples_leaf: %d" %best_est.min_samples_leaf)
# print ("max_features: %s" %best_est.max_features)
# print ("subsample: %.3f" %best_est.subsample)
# print ("Train R-squared: %.2f" %best_est.score(X_train,Y_train))
###################################################################

##### ERROR #####
# Encontra o MSE, que será o benchmark para este algoritmo, para identificar
# quão boa foi sua aproximação
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2_score = r2_score(Y_pred, Y_test)
cv_scores = model_selection.cross_val_score(gbt, X, Y, cv=5)

print("------- ACCURACY ASSESSMENT -------")
print("RMSE: {:.4f}" .format(rmse))
print("R^2 score: {:.4f}" .format(r2_score))
print("CV Scores: {:.2f} (+/- {:.2f})" .format(cv_scores.mean(),\
                                               cv_scores.std() * 2))

##### PLOTS #####
# Plot outputs using scatter. Ticks are diabled and everything else is the clea-
# nest that I could. The 1st graph - Featura Importances normalized with the
# highest value. Useful function here is print ("Feature Importances")
plot_FeatureImportances(gbt, X, Y_test, Y_pred)
