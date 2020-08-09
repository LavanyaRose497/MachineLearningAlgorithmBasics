# COMPLETED
# LIBRARIES
from sklearn.neural_network import MLPRegressor, MLPClassifier
from  sklearn.model_selection import GridSearchCV
import pandas as pd

# print(MLPRegressor()) - to know the hyperparameters
# print(MLPClassifier())

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


# HYPER PARAMETER TUNING
mlp = MLPClassifier()
# LEARNING RATE, ACTIVATION FUNCTION AND NUMBER OF HIDDEN LAYERS - HYPERPARAMETERS
parameters = {'hidden_layer_sizes':[(10,), (50,), (100,)], # first parameter- no. of nodes in hidden layer,
              # 2nd parameter in tuple - no. of layers, when left blank - refers to 1.
              'activation':['relu', 'tanh', 'logistic'], # relu : neg. value - 0, pos. value - no effect
              # tanh: from -1 to 1, logistic: from 0 to 1 curve (sigmoid)
              'learning_rate':['constant', 'invscaling', 'adaptive']} # constant - takes initial LR and keeps it same throughout
                    # optimization process, inverse scaling - gradualy decreases LR at each step (large steps at first and slows
                    # down on nearing optimum, adaptive - LR constant until training loss decreases. once it stops decreasing, it
                    # will take smaller steps
cv = GridSearchCV(mlp, parameters, cv=5)
cv.fit(X, y)
print_Results(cv)

print("Best Estimator: ", cv.best_estimator_)
# CREATE OBJ AND FIT