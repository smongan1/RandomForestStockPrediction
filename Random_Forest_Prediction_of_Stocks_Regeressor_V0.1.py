# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:31:04 2021

@author: smong
"""

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

print('The sklearn version is: ' + sklearn_version) #The sklearn version is: 0.23.2

RF_pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestRegressor(random_state=47)
)

ticker_df_100 = pd.read_csv('Data_Points.csv')

print(ticker_df_100.head())

X_train, X_test, y_train, y_test = train_test_split(ticker_df_100.drop(columns=['next_day_change','next_month_change','date']), 
                                                    ticker_df_100.next_month_change.astype(float), test_size=0.2, 
                                                    random_state=47)

n_est = [int(n) for n in np.logspace(start=1, stop=3, num=20)]
grid_params = {
        'randomforestregressor__n_estimators': [483],
        'standardscaler': [StandardScaler(), None],
        'simpleimputer__strategy': ['mean'],
        'randomforestregressor__bootstrap': [True],
        'randomforestregressor__max_depth': [5],
        'randomforestregressor__max_features': ['auto'],
        'randomforestregressor__min_samples_leaf': [20],
        'randomforestregressor__min_samples_split': [2]
}


rf_grid_cv = GridSearchCV(RF_pipe, param_grid=grid_params, cv=5, n_jobs=-1)
rf_grid_cv.fit(X_train, y_train)
rf_best_cv_results = cross_validate(rf_grid_cv.best_estimator_, X_train, y_train, cv=5)

plt.subplots(figsize=(10, 5))
imps = rf_grid_cv.best_estimator_.named_steps.randomforestregressor.feature_importances_
rf_feat_imps = pd.Series(imps, index=X_train.columns).sort_values(ascending=False)[0:20]
rf_feat_imps.plot(kind='bar')
plt.xlabel('features')
plt.ylabel('importance')
plt.title('Best random forest regressor feature importances');

rf_best_scores = rf_best_cv_results['test_score']

print(np.mean(rf_best_scores), np.std(rf_best_scores))
rf_neg_mae = cross_validate(rf_grid_cv.best_estimator_, X_train, y_train, 
                            scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)

rf_mae_mean = np.mean(-1 * rf_neg_mae['test_score'])
rf_mae_std = np.std(-1 * rf_neg_mae['test_score'])
print(rf_mae_mean, rf_mae_std)
print(mean_absolute_error(y_test, rf_grid_cv.best_estimator_.predict(X_test)))
print(rf_grid_cv.best_params_)