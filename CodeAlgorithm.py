#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:06:05 2024

@author: rabbiyayounas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Read the data

df = pd.read_csv(r"Fuel_Consumption_Ratings.csv")

# Data Exploration

df.head()
#describe() returns the summary for numeric columns.
#transpose()) is used to transpose the DataFrame, switching rows and columns. This can make the output easier to read

df.describe().T

#df.isnull().any() function in pandas is used to identify columns in a DataFrame that contain missing values
df.isnull().any()

# four heading of the data 
df_ = df[['Engine Size','Cylinders','Fuel Consumption', 'CO2 Emissions']]
df_.head()

#The df.hist() method in pandas is used to create histograms for each numeric column in a DataFrame. 
df_.hist()
plt.show()

# plot each of these features:

plt.scatter(df_['Fuel Consumption'], df_['CO2 Emissions'], color='red')
plt.xlabel("Fuel Consumption")
plt.ylabel("CO2 Emissions")
plt.show()

plt.scatter(df_['Engine Size'], df_['CO2 Emissions'], color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

plt.scatter(df_['Cylinders'], df_['CO2 Emissions'], color='black')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.show()

X = df_[['Fuel Consumption']]
y = df_[['CO2 Emissions']]

# Modelling

reg_model = DecisionTreeRegressor().fit(X, y)

#eaborn's regplot function to create a scatter plot with a regression line
#scatter_kws={'color': 'b', 's':9}: Customizes the appearance of the scatter plot points.
#color='b': Sets the color of the scatter points to blue.
#s=9: Sets the size of the scatter points.
#ci=False: Disables the confidence interval around the regression line.
#color='r': Sets the color of the regression line to red.

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's':9},
                 ci=False, color='r')  #güven aralığı false, yani ekleme
# g.set_title(f'Model Equation: CO2 Emissions = {round(reg_model.intercept_[0], 2)} + Fuel*{round(reg_model.coef_[0][0], 2)}')

#using seaborn's regplot (or any other plotting function), you can customize the labels of the axes using the set_ylabel and set_xlabel methods of the returned plot object.
g.set_ylabel('CO2 Emissions')
g.set_xlabel('Fuel Consumption')
plt.show()

# Evaluation of the model

# MSE
y_pred = reg_model.predict(X)

#the actual target values (y) and the predicted values (y_pred) 
mean_squared_error(y, y_pred)

#used numpy, 
# 229.99811521450823
y.mean() #260.11
y.std() #64.78

# RMSE [Root Mean Squared Error (RMSE)] using sing scikit-learn's mean_squared_error and numpy's sqrt function,
np.sqrt(mean_squared_error(y, y_pred))
# 15.165688748438306

# MAE , Mean Absolute Error (MAE) between the actual target values (y) and the predicted values (y_pred) using scikit-learn's mean_absolute_error function,
mean_absolute_error(y, y_pred) #to see the accuracy 
# 6.258727993638971

# R-SQUARED
reg_model.score(X, y)
# 0.9451350821741584



df = pd.read_csv(r"Fuel_Consumption_Ratings.csv")
df.head()
df.describe().T
df_ = df[['Engine Size','Cylinders','Fuel Consumption', 'CO2 Emissions']]
df_.head()
df_.describe().T


#Panda function , X = df_.drop('CO2 Emissions', axis=1): This line drops the column named 'CO2 Emissions'
X = df_.drop('CO2 Emissions', axis=1)   # drop the co2 emission variable from X
y = df_[["CO2 Emissions"]]
df_.head()


# Creating train and test data
# Around 80% of the entire data will be used for training and 20% for testing.
#using train_test_split from scikit-learn
#test_size=0.20: Specifies that 20% of the data should be used for testing, and the remaining 80% for training.
#random_state=1: Sets a seed for reproducibility, ensuring that the split is the same each time you run the code.


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
y_test.shape
y_train.shape
reg_model = DecisionTreeRegressor().fit(X_train, y_train)

# # The coefficients
# print ('Intercept: ',reg_model.intercept_)    #21.96963797
# print ('Coefficients: ', reg_model.coef_[0])  #1.12060648,  3.17912023, 19.4271568



#confusion 
new_data = [[8.00], [16.00], [26.10]]
new_data_2 = pd.DataFrame(new_data).T


#predictions = reg_model.predict(new_data_2): Uses the predict method of your trained DecisionTreeRegressor model (reg_model) to predict outcomes for new_data_2.
reg_model.predict(new_data_2)


# Evaluation of the model

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))  # 15.446625976286821 #the actual target values (y_train) 

# TRAIN RSQUARED , reg_model.score(X_train, y_train): This method computes the 𝑅Squraescore of the model on the training data (X_train, y_train).
reg_model.score(X_train, y_train) # 0.9423805674075392

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  # 11.090925681387699

# TEST RSQUARED , r2 score 
reg_model.score(X_test, y_test)  # 0.972015617095251


#reg_model: Your trained regression model (DecisionTreeRegressor in your case).
#X: Features DataFrame or array.
#y: Target variable DataFrame or array.
#cv=10: Number of folds for cross-validation. Here, it's set to 10-fold cross-validation.
#scoring="neg_mean_squared_error": Evaluation metric used for scoring.


np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))



df = pd.read_csv(r"Fuel_Consumption_Ratings.csv")
df.head()
df.describe().T
df1_ = df[['Engine Size','Cylinders','Fuel Consumption City', 'Fuel Consumption Hwy', 'CO2 Emissions']]
print(df1_.head())
df1_.describe().T

X = df1_.drop('CO2 Emissions', axis=1)   # drop the co2 emission variable from X
y = df1_[["CO2 Emissions"]]
df1_.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print(y_test.shape)
print(y_train.shape)
reg_model = DecisionTreeRegressor().fit(X_train, y_train)

# The coefficients
# print ('Intercept:',reg_model.intercept_)    #21.59204887
# print ('Coefficients:', reg_model.coef_[0])  #1.0463683,  3.23835864, 10.53429237, 8.97245603

#reg_model.predict(new_data_2): Predicts the target variable for new data
new_data_2 = [[8.00], [16.00], [30.30], [20.90]]
new_data_2 = pd.DataFrame(new_data_2).T
reg_model.predict(new_data_2)

# print(t)
# Evaluation of the model

# Train RMSE

# Computes the Root Mean Squared Error (RMSE) on the training set (X_train, y_train). Lower values indicate a better fit of the model to the training data.
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))  # 15.448257728971507

# TRAIN RSQUARED
u=reg_model.score(X_train, y_train) # 0.9423683931463056

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  # 11.043899812345115
print(u)
# TEST RSQUARED
p=reg_model.score(X_test, y_test)  # 0.9722524232995987

print(p)