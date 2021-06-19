import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
l1=[]
class LogisticRegression:
    lr=LogisticRegression()
    def fit(self):
        df=pd.read_csv("Diabetes.csv")
        print(df.info())
        self.x=df.iloc[:,:8]
        self.y=df['Outcome']
        self.xtrain,self.xtest,self.ytrain,self.ytest=train_test_split(self.x,self.y,test_size=0.22)
        self.lr.fit(self.xtrain,self.ytrain)
    def predict(self):
        self.pred=self.lr.predict(self.xtest)
        print(self.ytest)
        print(self.pred)
        print("Predictions have been done")
        print("Accuracy Score is :",metrics.accuracy_score(self.ytest,self.pred))
        print("cm is\n",metrics.confusion_matrix(self.ytest,self.pred))
        print("REPORTS Are\n",metrics.classification_report(self.ytest,self.pred))
        self.logistic=metrics.accuracy_score(self.ytest,self.pred)
        l1.append(self.logistic)
class DecisionTree:
    def fit(self):
        self.dr=DecisionTreeClassifier()
        df=pd.read_csv("Diabetes.csv")
        print(df.info())
        self.x=df.iloc[:,:8]
        self.y=df['Outcome']
        self.xtrain,self.xtest,self.ytrain,self.ytest=train_test_split(self.x,self.y,test_size=0.22)
        self.dr.fit(self.xtrain,self.ytrain)
    def predict(self):
        self.pred=self.dr.predict(self.xtest)
        print(self.ytest)
        print(self.pred)
        print("Predictions have been done")
        print("Accuracy Score is :",metrics.accuracy_score(self.ytest,self.pred))
        print("cm is\n",metrics.confusion_matrix(self.ytest,self.pred))
        print("REPORTS Are\n",metrics.classification_report(self.ytest,self.pred))
        self.decision=metrics.accuracy_score(self.ytest,self.pred)
        l1.append(self.decision)
class Knn:
    def fit(self):
        n=int(input("Enter the no of neighbors you want:"))
        self.clf=KNeighborsClassifier(n_neighbors=n)
        df=pd.read_csv("diabetes.csv")
        self.x=df.iloc[:,:8]
        self.y=df['Outcome']
        self.xtrain,self.xtest,self.ytrain,self.ytest=train_test_split(self.x,self.y,test_size=0.22)
        self.clf.fit(self.xtrain,self.ytrain)
        print("Machine has been trained")
    def predict(self):
        self.pred=self.clf.predict(self.xtest)
        print("Predictions have been done")
        print("Accuracy Score is\n",metrics.accuracy_score(self.ytest,self.pred))
        print("cm is\n",metrics.confusion_matrix(self.ytest,self.pred))
        print("reports\n",metrics.classification_report(self.ytest,self.pred))
        self.knn=metrics.accuracy_score(self.ytest,self.pred)
        l1.append(self.knn)
class Graph:
    def bar(self):
        print(l1)
        plt.bar(x=l1,color="y")
        plt.show()
