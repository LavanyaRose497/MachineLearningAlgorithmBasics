# COMPLETED
# SVM FITS A HYPERPLANE TO CLASSIFY DATA

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# IMPORT DATA
data = pd.read_csv(r"Dataset/svm_train_data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, 6].values
print(len(X), len(y))
print(X, y)


# FOR PRINTING RESULTS
def print_Results(results):
    print("Best Parameters are: {} ".format(results.best_params_))
    mean = results.cv_results_['mean_test_score']
    std = results.cv_results_['std_test_score']
    for m, s, p in zip(mean, std, results.cv_results_['params']):
        print('{} (+/- {}) for {}'.format(round(m, 3), round(s*2, 3), p))


# CREATE OBJECT FOR SVM AND FIT IN GRIDSEARCH WITH K-FOLD CROSSVALIDATION
svc = SVC()
# C AND KERNEL ARE HYPERPARAMETERS, WHCIH ARE ADJUSED TO GET A GOOD MODEL TO FIT
# C IS LIKE A PENALTY, IF C IS HIGH, HIGH PENALTY FOR MISCLASSIFICATION SO MODEL CLASSIFIES EACH AND EVERY POINT CORRECTLY(OVERFIT)
# IF C IS TO LOW, THEN MODEL THINKS I AM DOING GOOD AND GOES ON CLASSIFYING WRONGLY(UNDERFIT)

# SO BEST PARAMETERS ARE FOUND HERE
parameters = {'kernel':['linear', 'rbf'], 'C':[.1, 1, 10]}
cv = GridSearchCV(svc, parameters, cv = 5)
cv.fit(X, y)
print_Results(cv)

print(cv.best_estimator_)