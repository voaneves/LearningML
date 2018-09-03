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

##### DECLARING AND VALIDATING #####
# While loading this dataset, the date field needs to be divided into subcompo-
# nents. The heatmap below is to verifying whether there are NaN values. If so,
# we need to 'fillna' them
current_folder = os.path.join(os.path.dirname(__file__),\
                              'Input\energydata_complete.csv')
df = pd.read_csv(current_folder, parse_dates = ['date'], encoding = "utf8")

df['month'] = df['date'].dt.month # Feature enginnering: Month
df['monthday'] = df['date'].dt.day # Feature enginnering: Day
df['weekday'] = df['date'].dt.weekday # Feature enginnering: Day of the Week
df['time'] = df['date'].dt.time # Feature enginnering: Time
df['totalCons'] = df['Appliances'] + df['lights']

# sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
# plt.tight_layout()
# plt.show()

# ##### ANALYSING THE DATA VISUALLY #####
# # In order to properly understand the data at hand, we need to visualize how
# # it's distributed and what we have to focus on. So the graphs below are meant
# # for this purpose
# def sumData(df, colSearch, colSum):
#     """Return x and y which can be used to plot graphs.
#     This function counts occurrences"""
#     aggregated = df.reset_index().groupby(by=[colSearch]).agg({colSum:'sum'})
#     aggregated[colSearch] = aggregated.index
#
#     x = aggregated[colSearch] # These are the labels (x)
#     y = aggregated[colSum] # These are the sum (y)
#
#     return x, y
#
# x_month, y_month = sumData(df, 'month', 'totalCons')
# x_day, y_day = sumData(df, 'monthday', 'totalCons')
# x_weekday, y_weekday = sumData(df, 'weekday', 'totalCons')
# x_time, y_time = sumData(df, 'time', 'totalCons')
#
# pos_month = np.arange(x_month.shape[0]) + .5
# pos_day = np.arange(x_day.shape[0]) + .5
# pos_weekday = np.arange(x_weekday.shape[0]) + .5
# pos_time = np.arange(x_time.shape[0]) + .5
#
# plt.style.use('ggplot')
#
# plt.subplot(2, 2, 1)
# plt.title('Soma por mês')
# plt.bar(pos_month, y_month, align='center')
# plt.xticks(pos_month, x_month)
# plt.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     length=0   )       # labels along the bottom edge are off
#
# plt.subplot(2, 2, 2)
# plt.title('Soma por dia')
# plt.bar(pos_day, y_day, align='center')
# plt.xticks(pos_day, x_day)
# plt.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     length=0   )       # labels along the bottom edge are off
#
# plt.subplot(2, 2, 3)
# plt.title('Soma por dia da semana')
# plt.bar(pos_weekday, y_weekday, align='center')
# plt.xticks(pos_weekday, x_weekday)
# plt.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     length=0   )       # labels along the bottom edge are off
#
# plt.subplot(2, 2, 4)
# plt.title('Soma por horário')
# plt.bar(pos_time, y_time, align='center')
# plt.xticks(pos_time, x_time)
# plt.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     length=0   )       # labels along the bottom edge are off
# plt.gcf().autofmt_xdate()
#
# plt.tight_layout()
# plt.show()

def GradientBooster(param_grid, n_jobs):
    """Function to optimize the fit with data. Start using:
    1. min_samples_split = 0.5%-1% total values
    2. min_samples_leaf = 0.1*min_samples_split (or round to 1)
    3. max_depth = should be chosen from 5-8
    4. max_features = 'sqrt'
    5. subsample = 0.8 (common value to start)
    """
    estimator = ensemble.GradientBoostingRegressor(n_estimators = 900,\
                                                   max_depth = 10,\
                                                   min_samples_split = 100,\
                                                   min_samples_leaf = 8,\
                                                   learning_rate = 0.1,\
                                                   subsample = 0.8,\
                                                   max_features =  'sqrt',\
                                                   random_state = 42)
    classifier = model_selection.GridSearchCV(estimator=estimator, cv=5,\
                                              param_grid=param_grid,\
                                              n_jobs=n_jobs)
    classifier.fit(X_train, Y_train)
    print ("Best Estimator learned through GridSearch")
    print (classifier.best_estimator_)
    return classifier.best_estimator_

# É necessária então a divisão dos datasets, pelo método train_test_split. Para
# encontrar o tamanho de cada tensor que foi dividido, print(X_train.shape)
X = df.drop(['date', 'Appliances', 'lights', 'month', 'monthday', 'weekday',\
            'totalCons','time'], axis=1)
Y = df['totalCons']
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,\
                                   test_size = 0.1, random_state = 42)

# ###################################################################
# ##### FOR FINDING THE OPTIMAL VALUES, USING GradientBooster() #####
# # Order for tuning parameters:
# # 1. Tune n_estimators with learning_rate=0.1
# # 2. Tune max_depth and num_samples_split
# # 3. Tune min_samples_leaf
# # 4. Tune max_features
# # 5. Tune subsample
# # 6. Decrease learning_rate while increasing n_estimators proportionally
# param_grid = {'subsample': [0.4, 0.6, 0.8, 0.9]#,
#               #'min_samples_split': [100, 150, 200]
#              }
# n_jobs = 1
# best_est = GradientBooster(param_grid, n_jobs)
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
# ##################################################################
#

##### MODEL #####
# O método para fit dos dados do Boston serão os de Linear Regression
params = {'n_estimators': 900, 'max_depth': 15, 'min_samples_split': 100,\
         'min_samples_leaf': 8, 'learning_rate': 0.01, 'loss': 'ls',\
         'max_features': 'sqrt', 'subsample': 0.8, 'random_state': 42}
gbt = ensemble.GradientBoostingRegressor(**params)
gbt.fit(X_train, Y_train)

Y_pred = gbt.predict(X_test)

##### ERROR #####
# Encontra o MSE, que será o benchmark para este algoritmo, para identificar
# quão boa foi sua aproximação
mse = mean_squared_error(Y_test, Y_pred)
acc = gbt.score(X_test, Y_test)
cv_scores = model_selection.cross_val_score(gbt, X, Y, cv=5)

print("Model Accuracy: {:.4f}" .format(acc))
print("MSE: {:.4f}" .format(mse))
print("CV Scores: {:.2f} (+/- {:.2f})" .format(cv_scores.mean(),\
                                               cv_scores.std() * 2))

##### PLOTS #####
# Plot outputs using scatter. Ticks are diabled and everything else is the clea-
# nest that I could. The 1st graph - Featura Importances normalized with the
# highest value. Useful function here is print ("Feature Importances")
feature_importances = gbt.feature_importances_
feature_importances = 100*(feature_importances/feature_importances.max())
pos = np.arange(feature_importances.shape[0]) + .5
labels_X = X.columns.values

# It's important to use np.array before argsort, if it's not an array
idx = np.argsort(feature_importances)

feature_importances = np.array(feature_importances)[idx]
labels_X = np.array(labels_X)[idx]

plt.figure(figsize=(14, 6))
plt.style.use('ggplot')
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
plt.scatter(Y_test, Y_pred, alpha=0.75, label='Comparative Prices')
plt.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    length=0   )       # labels along the bottom edge are off
legend = plt.legend(loc='upper left', frameon=True, handletextpad=0.1)
legend.get_frame().set_facecolor('white')
plt.xlabel("Real prices")
plt.ylabel("Predicted prices")
plt.title("Predicted prices vs Real Prices")
plt.show()
