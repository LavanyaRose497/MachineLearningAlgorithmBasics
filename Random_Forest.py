# COMPLETED
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

# IMPORT DATA
data = pd.read_csv(r"Dataset/svm_train_data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, 6].values
# print(len(X), len(y))
# print(X, y)

# FOR PRINTING RESULTS
def print_Results(results):
    print("Best Parameters are: {} \n".format(results.best_params_))
    mean = results.cv_results_['mean_test_score']
    std = results.cv_results_['std_test_score']
    for m, s, p in zip(mean, std, results.cv_results_['params']):
        print('{} (+/- {}) for {}'.format(round(m, 3), round(s*2, 3), p))

# HYPERPARAMTERS: n_estimators-controls how many individual trees are built, max_depth-controls the depth of individual tree
rf = RandomForestClassifier()
parameters = {'n_estimators':[5, 50, 250], 'max_depth':[2, 4, 8, 16, 32, None]}
cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(X, y)
print_Results(cv)
print("Best Estimator: ", cv.best_estimator_)