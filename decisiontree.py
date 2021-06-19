import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv("Decisiontree.csv")
le=LabelEncoder()
clf=DecisionTreeClassifier()
df['Age']=le.fit_transform(df['Age'].values.tolist())
df['Income']=le.fit_transform(df['Income'].values.tolist())
df['Gender']=le.fit_transform(df['Gender'].values.tolist())
df['Marital_Status']=le.fit_transform(df['Marital_Status'].values.tolist())
print(df)
x=df[['Age','Income','Gender','Marital_Status']]
y=df['Buys']
##xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.22)
##clf.fit(xtrain,ytrain)
##ypred=clf.predict(xtest)
##print("This is ytest\n",ytest)
##print("This is ypred\n",ypred)
##print(metrics.accuracy_score(ytest,ypred))
clf.fit(x,y)
ypred=clf.predict([[2,2,0,1]])
print(ypred)
