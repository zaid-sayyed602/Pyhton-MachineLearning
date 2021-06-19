from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
df=pd.read_csv("cardetails.csv")
print(df.isna().sum())
df1=df.iloc[:,1::]
print(df1)
x=df1.loc[:,[("year"),("fuel"),("seller_type"),("transmission"),("owner")]]
y=df["selling_price"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
obj=LinearRegression()
obj.fit(xtrain,ytrain)
