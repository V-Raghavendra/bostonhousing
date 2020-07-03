#importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sks
import seaborn as sns
import math
import yellowbrick

#loading the dataset
from sklearn.datasets import load_boston
boston = load_boston()

#input variable
x = boston.data
#output variable
y = boston.target

#spliting the data to train and split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

#print the dimesions of the dataset
print("x_train shape :", x_train.shape)
print("x_test shape : ", x_test.shape)
print("y_train shape :",y_train.shape)
print("y_test shape :", y_test.shape)


#applying  regression to the datset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the output
y_pred = regressor.predict(x_test)
print("Look Ytest" , y_test)
print("Look Ypred" , y_pred)

#plotting
plt.scatter(x = y_test,y = y_pred, c='Blue',)
plt.show()

#visulaizing the model fit
from yellowbrick.regressor import PredictionError, ResidualsPlot
visualizer = PredictionError(regressor).fit(x_train, y_train)
visualizer.score(x_test, y_test)
visualizer.poof()

#evaluating the model with r2 score
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))



#Part 2 (DF)
print(boston)
df = pd.DataFrame(boston.data)
print("df\n", df)
df.columns = boston.feature_names
print("1" , df.head(10))
df["price"] = boston.target
print("2"  ,df.head())

#heatmap
Var_Corr = df.corr()
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)

# transform the features to higher degree features.
x_train_poly = poly_features.fit_transform(x_train)

# fit the transformed features to Linear Regression
poly_model = LinearRegression()

poly_model.fit(x_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(x_train_poly)

# predicting on test data-set
y_test_predicted = poly_model.predict(poly_features.fit_transform(x_test))

y_train_residual = y_train_predicted - y_train
y_test_residual = y_test_predicted - y_test

r2_train = r2_score(y_train, y_train_predicted)
print("R2 score of training set is {}".format(r2_train))

r2_test = r2_score(y_test, y_test_predicted)
print("R2 score of test set is {}".format(r2_test))

plt.scatter(x = y_test, y = y_pred, c='Green')
plt.show()























