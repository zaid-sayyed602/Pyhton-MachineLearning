import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
df=pd.read_csv("Mall.csv")
print(df.info())
#####################################
sb.countplot(x=df["Age"],palette="gist_earth_r")
plt.title("COUNT OF AGE")
plt.show()
###################################
sb.countplot(x=df["Annual Income (k$)"],palette="nipy_spectral")
plt.title("COUNT OF Annual Income")
plt.show()
###################################
sb.countplot(x=df["Spending Score (1-100)"],palette="Wistia_r")
plt.title("COUNT OF SPENDING SCORE")
plt.show()
###################################
sb.pairplot(df)
plt.show()
###################################
sb.jointplot(x=df["Age"],y=df["Annual Income (k$)"],hue=df["Gender"],palette="winter")
plt.title("AGE VS ANNUAL INCOME")
plt.show()
###################################
plt.title("GENDER VS SPENDING SCORE")
sb.boxenplot(x=df["Gender"],y=df["Spending Score (1-100)"])
plt.show()
###################################
plt.title("GENDER VS SPENDING SCORE")
sb.stripplot(x=df["Gender"],y=df["Spending Score (1-100)"],palette="ocean")
plt.show()
####---------------------------------####
le=LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"].tolist())
x=df.iloc[:,1:]
print(x)
####---ELBOW METHOD---###
l1=[]
for i in range(1,11):
    cl=KMeans(n_clusters=i)
    cl.fit(x)
    l1.append(cl.inertia_)
print(l1)
plt.plot(range(1,11),l1)
plt.xticks(range(1,11))
plt.show()
###----FITTING----###
cl=KMeans(n_clusters=5)
cl.fit(x)
centroid=cl.cluster_centers_
labels=cl.labels_
print(centroid)
plt.scatter(df['Annual Income (k$)'],df["Spending Score (1-100)"],c=labels)
plt.scatter(centroid[:,2],centroid[:,3],marker="*")
plt.show()

