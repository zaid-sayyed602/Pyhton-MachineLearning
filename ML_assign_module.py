import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics


df=pd.read_csv("creditcard.csv")
print(df.head())
print(df.describe())
print(df.info())
l1=[]
class Logistic:
    lr=LogisticRegression()
    def fit1(self):
        df=pd.read_csv("creditcard.csv")
        split=StratifiedShuffleSplit(test_size=0.3,random_state=42)
        for test_set,train_set in split.split(df,df["Class"]):
            train=df.loc[train_set]
            test=df.loc[test_set]
        self.xtrain=train.iloc[:,1:30]
        self.ytrain=train["Class"]
        self.xtest=test.iloc[:,1:30]
        self.ytest=test["Class"]
        self.lr.fit(self.xtrain,self.ytrain)
    def predict1(self):
        self.pred=self.lr.predict(self.xtest)
        print(self.ytest)
        print(self.pred)
        print("Predictions have been done")
        print("Accuracy Score is :",metrics.accuracy_score(self.ytest,self.pred))
        print("cm is\n",metrics.confusion_matrix(self.ytest,self.pred))
        print("REPORTS Are\n",metrics.classification_report(self.ytest,self.pred))
        self.logistic=metrics.accuracy_score(self.ytest,self.pred)
        a=self.logistic*100
        l1.append(a)
class Decision:
        def fit1(self):
            self.dt=DecisionTreeClassifier()
            df=pd.read_csv("creditcard.csv")
            split=StratifiedShuffleSplit(test_size=0.3,random_state=42)
            for test_set,train_set in split.split(df,df["Class"]):
                train=df.loc[train_set]
                test=df.loc[test_set]
            self.xtrain=train.iloc[:,1:30]
            self.ytrain=train["Class"]
            self.xtest=test.iloc[:,1:30]
            self.ytest=test["Class"]
            self.dt.fit(self.xtrain,self.ytrain)
        def predict1(self):
            self.pred=self.dt.predict(self.xtest)
            print(self.ytest)
            print(self.pred)
            print("Predictions have been done")
            print("Accuracy Score is :",metrics.accuracy_score(self.ytest,self.pred))
            print("cm is\n",metrics.confusion_matrix(self.ytest,self.pred))
            print("REPORTS Are\n",metrics.classification_report(self.ytest,self.pred))
            self.decision=metrics.accuracy_score(self.ytest,self.pred)
            a=self.decision*100
            l1.append(a)
class Knn:
    def fit1(self):
        df=pd.read_csv("creditcard.csv")
        split=StratifiedShuffleSplit(test_size=0.3,random_state=42)
        for test_set,train_set in split.split(df,df["Class"]):
            train=df.loc[train_set]
            test=df.loc[test_set]
        self.xtrain=train.iloc[:,1:30]
        self.ytrain=train["Class"]
        self.xtest=test.iloc[:,1:30]
        self.ytest=test["Class"]
        res=[]
##        for i in range(1,50):
##            self.kn=KNeighborsClassifier(n_neighbors=i,algorithm="kd_tree")
##            self.xtrain_check=self.xtrain.iloc[:20000]
##            self.ytrain_check=self.ytrain.iloc[:20000]
##            self.xtest_check=self.xtest.iloc[:6000]
##            self.ytest_check=self.ytest.iloc[:6000]            
##            self.kn.fit(self.xtrain_check,self.ytrain_check)
##            self.ypred=self.kn.predict(self.xtest_check)
##            res.append(np.mean(self.ypred != self.ytest_check))
##        plt.plot(range(1,50),res,marker="o")
##        plt.xticks(range(1,50))
##        plt.show()        
        self.n=int(input("Enter the no of neighbors you want\n"))
        self.knn=KNeighborsClassifier(n_neighbors=self.n,algorithm="kd_tree")
        self.knn.fit(self.xtrain,self.ytrain)
    def predict1(self):
        self.pred=self.knn.predict(self.xtest)
        print("Predictions have been done")
        print("Accuracy Score is\n",metrics.accuracy_score(self.ytest,self.pred))
        print("cm is\n",metrics.confusion_matrix(self.ytest,self.pred))
        print("reports\n",metrics.classification_report(self.ytest,self.pred))
        self.knnacc=metrics.accuracy_score(self.ytest,self.pred)
        a=self.knnacc*100
        l1.append(a)
        
class RandomForest:
    def fit1(self):
        df=pd.read_csv("creditcard.csv")
        split=StratifiedShuffleSplit(test_size=0.3,random_state=42)
        for test_set,train_set in split.split(df,df["Class"]):
            train=df.loc[train_set]
            test=df.loc[test_set]
        self.xtrain=train.iloc[:,1:30]
        self.ytrain=train["Class"]
        self.xtest=test.iloc[:,1:30]
        self.ytest=test["Class"]
##        res=[]
##        for i in range(1,100):
##            self.xtrain_check=self.xtrain.iloc[:20000]
##            self.ytrain_check=self.ytrain.iloc[:20000]
##            self.xtest_check=self.xtest.iloc[:10000]
##            self.ytest_check=self.ytest.iloc[:10000]
##            self.rf=RandomForestClassifier(n_estimators=i)
##            self.rf.fit(self.xtrain_check,self.ytrain_check)
##            self.ypred=self.rf.predict(self.xtest_check)
##            res.append(np.mean(self.ypred != self.ytest_check))
##        plt.plot(range(1,100),res,marker="o")
##        plt.xticks(range(1,100))
##        plt.show()        
        self.n=int(input("Enter the no of estimators you want\n"))
        self.rfc=RandomForestClassifier(n_estimators=self.n)
        self.rfc.fit(self.xtrain,self.ytrain)
    def predict1(self):
        self.pred=self.rfc.predict(self.xtest)
        print("Predictions have been done")
        print("Accuracy Score is\n",metrics.accuracy_score(self.ytest,self.pred))
        print("cm is\n",metrics.confusion_matrix(self.ytest,self.pred))
        print("reports\n",metrics.classification_report(self.ytest,self.pred))
        self.rfcacc=metrics.accuracy_score(self.ytest,self.pred)
        a=self.rfcacc*100
        l1.append(a)
class Graph:
    def visualization(self):
        print(l1)
        name=["LOGISTIC","DECISION","KNN","RANDOMFOREST"]
        plt.bar(name,l1)
        plt.show()
        
