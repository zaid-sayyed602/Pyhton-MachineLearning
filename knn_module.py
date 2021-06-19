import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier   
class Diabetes:
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
    def manvalue(self):
        n1=int(input("Pregnancies"))
        n2=int(input("Glucose"))
        n3=int(input("BloodPressure"))
        n4=float(input("SkinThickness"))
        n5=float(input("Insulin"))
        n6=float(input("BMI"))
        n7=float(input("DiabetesPedigreeFunction"))
        n8=int(input("Age"))
        d={"Pregnancies":[n1],"Glucose":[n2],"BloodPressure":[n3],
        "SkinThickness":[n4],"Insulin":[n5],"BMI":[n6],"DiabetesPedigreeFunction":[n7],"Age":[n8]}
        df1=pd.DataFrame(d)
        manpred=self.clf.predict(df1)
        print("Prediction on Outcome after seeing the input is:",manpred)
