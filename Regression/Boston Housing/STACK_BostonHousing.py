#!/usr/bin/env python
# Title         : STACK_BostonHousing.py
# Description   : Stacking was the natural progression of our algorithms trial.
#                 In here, we'll use prediction from a number of models in order
#                 to improve accuracy as it add linearly independent data to our
#                 dataset. Here we also use voting ensembler, using the best es-
#                 timator three timers on the stack of second level models.
#                 We'll find CV scores of each model on train_test_split then
#                 stack the models on a 5-KFold of the data, finding final CV
#                 score. We'll also plot the comparative graph of Real Prices vs
#                 Predicted Prices
# Author        : Neves4
# Outputs       : Figure with one plot      : 'Real Prices vs Predicted prices'
#                 Values                    : SVR CV Scores: 0.6798 (+/- 0.0895)
#                                             XGB CV Scores: 0.8784 (+/- 0.0598)
#                                             RF CV Scores: 0.8601 (+/- 0.0789)
#                                           STACK CV Scores: 0.8809 (+/- 0.0864)
# License       : MIT License
#==============================================================================

##### IMPORTING #####
import numpy as np
import xgboost as xgb
from sklearn import datasets
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score

sns.set() # set seaborn style

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1,
                                                    random_state = 42)

# ##### 1ST LEVEL MODELS #####
# # ElasticNet - baseline model #0
# print("------- FITTING ElasticNet -------")
# en_mdl = ElasticNet(alpha = 5.2, l1_ratio = 0.5, random_state = 42)
# en_cv_scores = cross_val_score(en_mdl, X_train, Y_train, cv=5, scoring='r2')
# print("  DONE! CV Scores: {:.4f} (+/- {:.4f})" .format(en_cv_scores.mean(),\
#                                                        en_cv_scores.std() * 2))

# SVR - baseline model #1
print("------- FITTING SVR -------")
svr_mdl = SVR(kernel = 'linear', C = 0.11, epsilon = 0.011, gamma = 0.1)
svr_cv_scores = cross_val_score(svr_mdl, X_train, Y_train, cv=5, scoring='r2')
print("  DONE! CV Scores: {:.4f} (+/- {:.4f})" .format(svr_cv_scores.mean(),\
                                                       svr_cv_scores.std() * 2))

# XGBRegressor - baseline model #2
print("------- FITTING XGBRegressor -------")
xgb_mdl = xgb.XGBRegressor(learning_rate = 0.0503, n_estimators = 339,
                        max_depth = 5, min_child_weight = 2, gamma = 0.17,
                        subsample = 0.84, colsample_bytree = 0.85,
                        reg_alpha = 0.008, reg_lambda = 1.2,
                        scale_pos_weight = 1, seed = 42)
xgb_cv_scores = cross_val_score(xgb_mdl, X_train, Y_train, cv=5, scoring='r2')
print("  DONE! CV Scores: {:.4f} (+/- {:.4f})" .format(xgb_cv_scores.mean(),\
                                                       xgb_cv_scores.std() * 2))

# RandomForestRegressor - baseline model #3
print("------- FITTING RandomForestRegressor -------")
rf_mdl = RandomForestRegressor(n_estimators = 95, max_features = 'auto',
                           max_depth = 18, min_samples_split = 2,
                           min_samples_leaf = 1, bootstrap = True,
                           random_state = 42)
rf_cv_scores = cross_val_score(rf_mdl, X_train, Y_train, cv=5, scoring='r2')
print("  DONE! CV Scores: {:.4f} (+/- {:.4f})" .format(rf_cv_scores.mean(),\
                                                       rf_cv_scores.std() * 2))

class Ensemble(object):
    """Ensemble base_models on train data than fit/predict

    The object input is composed of 'n_splits', 'stacker' and list of
    'base_models'.

    The __init__ method self-assign the inputs.

    The fit_predict method divides the dataset in 'n_splits' then it loops
    trough ammount of 'base_models' fitting all splits and then averaging it on
    a new column in the end. In the end, predictions are made with these new
    columns.

    If sought the use of voting ensemble, the ammount of models passed on
    base_models can be repeated.
    """

    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, Y, T):
        X = np.array(X)
        Y = np.array(Y)
        T = np.array(T)

        # Create folds on the dataset based on n_splits
        folds = list(KFold(n_splits = self.n_splits, shuffle = True,
                     random_state = 42).split(X, Y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        # Loop trough base_models
        print("------- FITTING Stacker - 2nd level -------")
        for i, clf in enumerate(self.base_models):

            # Create a dummy to calculate predictions on all folds
            S_test_i = np.zeros((T.shape[0], self.n_splits))

            # Loop trough data folds
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                Y_train = Y[train_idx]
                X_holdout = X[test_idx]
                Y_holdout = Y[test_idx]

                clf.fit(X_train, Y_train)
                Y_pred = clf.predict(X_holdout)[:]

                print ("  Model {}, fold {}. R^2 score: {:.4f}"\
                       .format(i, j, r2_score(Y_holdout, Y_pred)))

                S_train[test_idx, i] = Y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            # Update test data with average of predictions from the dummy
            S_test[:, i] = S_test_i.mean(axis = 1)

        # Print final CV score
        results = cross_val_score(self.stacker, S_train, Y, cv=5, scoring='r2')
        print("\033[1;92mDONE! \033[0;0m\033[1;37mCV scores: {:.4f} (+/- {:.4f})"
              .format(results.mean(), results.std() * 2))

        # After creating new features on the test data, fit the chosen stacker
        # on train data and finally predict on test data, then return
        self.stacker.fit(S_train, Y)
        final_prediction = self.stacker.predict(S_test)[:]

        return final_prediction

stack = Ensemble(n_splits = 5, stacker = svr_mdl,
                 base_models = (xgb_mdl, rf_mdl, xgb_mdl, svr_mdl, xgb_mdl))

stack_pred = stack.fit_predict(X_train, Y_train, X_test)

##### PLOTS #####
# Plot outputs using scatter. Ticks are diabled and everything else is the clea-
# nest that I could. Predicted prices vs Real Prices
custom_style = {'axes.labelcolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white'}
data = pd.DataFrame(data = {'stack_pred': stack_pred, 'Y_test': Y_test})
ax = sns.lmplot(x='Y_test', y='stack_pred', data = data, truncate=True, size=5)
ax.set_axis_labels("Real prices", "Predicted prices")
plt.tick_params(axis='both', colors='gray')
plt.title("Real vs Predicted prices on Boston Housing", fontweight = 'bold')
plt.tight_layout()
plt.show()
