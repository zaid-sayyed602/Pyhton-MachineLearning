import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
obj=LogisticRegression()
d={"age":[18,19,20,21,22,23,24,25,26,27,28,29,30],"height":[5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,5.10,5.11,5.12,6.00],
   "size":["m","m","m","m","m","m","l","l","l","l","l","l","l"]}
df=pd.DataFrame(d)
print(df)
x=df.iloc[:,:2]
y=df["size"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.22)
obj.fit(xtrain,ytrain)
size_pred=obj.predict(xtest)
plt.scatter(xtest["age"],ytest,marker="o")
plt.scatter(xtest["age"],size_pred,marker="x")
plt.show()
