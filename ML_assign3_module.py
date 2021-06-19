import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC

class Knn: 
    def fit(self):
        n=int(input("Enter the no of neighbors you want:"))
        self.knn=KNeighborsClassifier(n_neighbors=n)
        df=pd.read_csv("iris.csv")
        self.x=df.iloc[:,:4]
        self.y=df["species"]
        self.xtrain,self.xtest,self.ytrain,self.ytest=train_test_split(self.x,self.y,test_size=0.2)
        self.knn.fit(self.xtrain,self.ytrain)
        print("Machine has been trained")
    def predict(self):
        self.pred=self.knn.predict(self.xtest)
        print("prediction has been done")
        print("Accuracy Score is :",metrics.accuracy_score(self.ytest,self.pred))
        print("cm is\n",metrics.confusion_matrix(self.ytest,self.pred))
        print("REPORTS Are\n",metrics.classification_report(self.ytest,self.pred))
    def graph(self):
        df=pd.read_csv("iris.csv")
        sb.scatterplot(x=df["sepal_length"],y=df["sepal_width"],hue=df["species"])
        plt.show()
    def manvalue(self):
        input1=int(input("Enter the sepal lenght\n"))
        input2=int(input("Enter the sepal width\n"))
        input3=int(input("Enter the petal lenght\n"))
        input4=int(input("Enter the petal width\n"))
        self.manpred=self.knn.predict([[input1,input2,input3,input4]])
        print("prediction made by the model is\n",self.manpred)
