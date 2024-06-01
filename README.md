Steps : 
1) Import all the important libararies :
   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

2) Read the data
3) Declare X and Y
   X = df_[['Fuel Consumption']]
   y = df_[['CO2 Emissions']]
4) Modelling
   reg_model = DecisionTreeRegressor().fit(X, y)
5) Evaluation of the model
