# Used_car_Machine_Learning
This project was created using Jupyter Notebook. In this project, I used different types of machine learning models, including gradient boosting, catboost, and lightgbm to predict used car prices from a dataset. I took the features from the dataset and the price column as the target to make those predictions using the Root Mean Squared Error metric to evaluate those machine learning models. Also, I used the %%time Magic Command in Jupyter Notebook to see the execution time of each machine learning model, being able to determine a machine learning model with a good execution time and good quality also.   

The libraries used in this project are:  
import pandas as pd  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split   
from sklearn.metrics import mean_squared_error, mean_absolute_error  
from matplotlib import pyplot as plt  
import numpy as np  
from sklearn.dummy import DummyRegressor  
from sklearn.model_selection import cross_val_score  
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier  
import seaborn as sns  
from sklearn.ensemble import GradientBoostingRegressor  
import lightgbm as lgb  
from lightgbm import LGBMRegressor  
from catboost import CatBoostRegressor  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import GridSearchCV  
 
