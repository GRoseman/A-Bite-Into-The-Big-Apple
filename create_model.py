from pandas import read_csv
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals.joblib import dump
from sklearn.model_selection import GridSearchCV, train_test_split

manhattan = read_csv('sales.csv')[['latitude', 'longitude', 'residential units', 'commercial units', 'lot', 'sale price']]
manhattan['sale price'] = (manhattan['sale price']/1000).astype(int)
manhattan = manhattan[(manhattan['sale price']>=250)&(manhattan['sale price']<=10000)]

param_grid = {'n_estimators': [35, 40, 45, 50], 'max_features': [2, 3, 4],
'min_samples_split': [2, 3, 4], 'max_depth': [20, 30, 40]}

reg = GridSearchCV(RandomForestRegressor(), param_grid, scoring='neg_median_absolute_error')
reg.fit(manhattan.drop(['sale price'], axis=1), manhattan['sale price'])

dump(reg.best_estimator_.fit(manhattan.drop(['sale price'], axis=1), manhattan['sale price']), 'website/model/model.pkl')