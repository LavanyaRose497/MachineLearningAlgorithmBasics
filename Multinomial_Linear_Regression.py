# PACKAGES
import pandas as pd
import sklearn.model_selection as sm
import sklearn.linear_model as sl
import sklearn.preprocessing as sp

# GET DATA
data = pd.read_csv("Dataset/multiple_linear_reg.csv")
X = data.iloc[:, :-1].values
Y = data.iloc[:, 4].values

# PREPROCESSING
# ENCODING THE INDEPENDENT CATEGORICAL VARIABLE
X_lab = sp.LabelEncoder()
X[:, 3] = X_lab.fit_transform(X[:, 3])

# TRAIN TEST SPLIT
X_train, X_test, Y_train, Y_test = sm.train_test_split(X, Y, test_size=0.3)
# print(X_train, Y_train)

# MODEL OBJECT
obj = sl.LinearRegression()

# FIT MODEL
obj.fit(X_train, Y_train)

# PREDICT TEST DATA
res = obj.predict(X_test)

for i in range(len(res)):
    print(res[i], Y_test[i])

# PREPROCESS RUN IME DATA
# PREDICT RUN TIME DATA

