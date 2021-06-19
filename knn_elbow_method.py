import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv("diabetes.csv")
x=df.iloc[:,:8]
y=df['Outcome']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)
res=[]
for i in range(1,50):
    clf=KNeighborsClassifier(n_neighbors=i)
    clf.fit(xtrain,ytrain)
    ypred=clf.predict(xtest)
    res.append(np.mean(ypred != ytest))
    ac=metrics.accuracy_score(ypred,ytest)
    print("for i neighbor :",i,"=>",ac)

print(res)
plt.plot(range(1,50),res,marker="o")
plt.xticks(range(1,50))
plt.show()
