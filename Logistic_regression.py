import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
df=pd.read_csv("titanic.csv")
print(df.info())
print(df.isna().sum())
df.drop("Cabin",inplace=True,axis=1)
df.drop("Name",inplace=True,axis=1)
df.drop("PassengerId",inplace=True,axis=1)
am=int(df["Age"].mean())
df["Age"]=df["Age"].fillna(am)
df["Embarked"]=df["Embarked"].fillna(method="bfill")
print(df.isna().sum())
x=df.iloc[:,1:9]
y=df["Survived"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.22)
obj=LogisticRegression()
obj.fit(xtrain,ytrain)
pred=obj.predict(xtest)
print("Training Successful")
print(ytest)
print(pred)
