# COMPLETED
# LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pl


# INPUT DATA
dataset = pd.read_csv('Dataset/linear_reg_data.csv')
x_data = dataset.iloc[:, :-1].values
y_data = dataset.iloc[:, 1].values
# print(x_data, "\n", y_data)

# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

# CREATE A MODEL OBJECT
obj = LinearRegression()

# FIT THE MODEL
obj.fit(x_train, y_train)

# PREDICT
result = obj.predict(x_test)

# PLOT THE GRAPH
pl.scatter(x_train, y_train, color = 'red')
pl.scatter(x_test, obj.predict(x_test), color = 'yellow')
pl.plot(x_test, obj.predict(x_test), color = 'yellow')
pl.title("Salary vs Experience")
pl.xlabel("Salary")
pl.ylabel("Experience")
pl.show()


























'''
import pandas as pd
import matplotlib.pyplot as mat

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# READ DATA
data = pd.read_csv("Dataset\\linear_reg_data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
# print(X,y)

# SPLIT INTO TRAINING AND TESTING DATA
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state= 0)

# CREATE OBJECT OF MODEL
linear_obj = LinearRegression()

# METHOD THAT FITS THE OBJECT TO THE TRAINING SET
linear_obj.fit(x_train, y_train)

# MAKE PREDICTIONS
result = linear_obj.predict(x_test)
print(result)

# VISUALIZATION
mat.scatter(x_train, y_train, color = 'red')
mat.plot(x_train, linear_obj.predict(x_train), color = 'green')
mat.scatter(x_test, y_test, color = "yellow")
mat.plot(x_test,linear_obj.predict(x_test), color = 'blue')
mat.title("Salary vs Experience")
mat.xlabel('Salary')
mat.ylabel('Exp')
mat.show()
'''