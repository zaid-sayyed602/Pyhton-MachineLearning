import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import metrics 
data = pd.read_csv("Svm_data.csv",header=None)
x=data.values[:, :2]
y=data.values[:, 2]

a=plt.scatter(x[:, 0], x[:, 1],c=y)
clf=SVC(kernel="linear",C=1)
clf_fit=clf.fit(x,y)

#geting limit value of axis

ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()

#mesh grid
xx=np.linspace(xlim[0],xlim[1],200)
yy=np.linspace(ylim[0],ylim[1],200)
print(xx.shape)
YY,XX=np.meshgrid(yy,xx)
print(XX.shape)
xy=np.vstack([XX.ravel(),YY.ravel()]).T
z=clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX,YY,z,colors='r',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
b=ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=100,facecolors='r')
plt.legend([b],['Support Vectors'])
plt.show()
acc=metrics.accuracy_score(clf.predict(x),y)
print(acc)
