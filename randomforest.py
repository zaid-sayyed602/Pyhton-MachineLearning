import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

df=pd.read_csv('iris.csv')
print(df.head())
print(df.info())
print(df.describe())
x=df.iloc[:,:4]
y=df["species"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
res=[]
##for i in range(1,120):
##    rf=RandomForestClassifier(n_estimators=i)
##    rf.fit(xtrain,ytrain)
##    ypred=rf.predict(xtest)
##    res.append(np.mean(ypred != ytest))
##    ac=metrics.accuracy_score(ypred,ytest)
##    print("for i estimators :",i,"=>",ac)
##plt.plot(range(1,120),res,marker="o")
##plt.xticks(range(1,120))
##plt.show()
rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)
ypred=rf.predict(xtest)
print(metrics.accuracy_score(ytest,ypred))
