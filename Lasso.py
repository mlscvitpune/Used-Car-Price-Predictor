#Importing the Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import metrics
import pickle

#Reading the Data
train_data = pd.read_csv("E:/ET-A/Hacktoberfest/Used-Car-Price-Predictor/final_train.csv")

#Data Analysis
print(train_data.head())
print(train_data.shape)
print(train_data.fuel_type.value_counts())
print(train_data.transmission.value_counts())
print(train_data.info())

#Checking Null values
print(train_data.isnull().sum())

#encoding categorical values
train_data.replace({'fuel_type':{'petrol':0,'diesel':1,'petrol & cng':2,'petrol & lpg':3}},inplace=True)
print(train_data.head())
train_data.replace({'transmission':{'manual':0,'automatic':1}},inplace=True)
print(train_data.body_type.value_counts())
train_data.replace({'body_type':{'hatchback':0,'sedan':1,'suv':2,'luxury suv':3,'luxury sedan':4}},inplace=True)
x = train_data.drop(['car_name','city'],axis=1)
y = train_data['sale_price']
print(x)
print(y)

#Splitting the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Modelling
Lasso_model = Lasso(alpha=0.01) 
Lasso_model.fit(x_train,y_train)

#Prediction
Prediction_train = Lasso_model.predict(x_train)
Prediction_test = Lasso_model.predict(x_test)

#Metrics
train_r2_error = metrics.r2_score(y_train,Prediction_train)
print("Train R squared Error: ",train_r2_error)


test_r2_error = metrics.r2_score(y_test,Prediction_test)
print("Test R squared Error: ",test_r2_error)
pred_train_lasso= Lasso_model.predict(x_train)
print(np.sqrt(metrics.mean_squared_error(y_train,pred_train_lasso)))
print(metrics.r2_score(y_train, pred_train_lasso))

pred_test_lasso= Lasso_model.predict(x_test)
print(np.sqrt(metrics.mean_squared_error(y_test,pred_test_lasso))) 
print(metrics.r2_score(y_test, pred_test_lasso))


pickle.dump(Lasso_model, open('Lasso_model.pkl', 'wb'))


