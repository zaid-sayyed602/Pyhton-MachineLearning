from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
d={}
y=[]
a=[]
choice=0
while(choice<3):
    print("1.to enter data in the csv file")
    print("2.graph")
    choice=int(input("Enter your choice:"))
    if(choice==1):       
        year=int(input("Enter the start year\n"))
        n=int(input("Enter the no of years data you want\n"))
        for i in range(n):
            year += 1
            age=2021-year
            y.append(year)
            a.append(age)
        d["year"]=y
        d["age"]=a     
        d=pd.DataFrame(d)
        d.to_csv("lr_age.csv",index=False)
        print("successfully created")
    elif(choice==2):
        d=pd.read_csv("lr_age.csv")
        sb.relplot(x=d["year"],y=d["age"],kind="line")
        plt.show()
        x=d["year"]
        y=d["age"]
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
        print(xtrain)
        print(xtest)
##      sb.jointplot(x=xtrain,y=ytrain,kind="reg")
##      sb.jointplot(x=xtest,y=ytest,kind="reg")
##      plt.show()
        xtrain=np.array(xtrain).reshape(-1,1)
        ytrain=np.array(ytrain).reshape(-1,1)
        xtest=np.array(xtest).reshape(-1,1)
        ytest=np.array(ytest).reshape(-1,1)
        obj=LinearRegression()
        obj.fit(xtrain,ytrain)
        age_pred=obj.predict(xtest)
        plt.scatter(x=xtest,y=ytest,marker="o",color="r")
        plt.scatter(x=xtest,y=age_pred,marker="x",color="g")
        plt.plot(xtest,age_pred)
        plt.show()
        #print("mean squared error is: ",mean_squared_error(ytest,age_pred))
        #print("mean squared error is: ",np.mean(age_pred-ytest)**2)
    elif(choice==3):
        print("Exited")
        break
