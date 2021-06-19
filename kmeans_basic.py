import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.DataFrame({'X':[0.1,0.15,0.08,0.16,0.2,0.25,0.24,0.3],
                 'y':[0.6,0.71,0.9,0.85,0.3,0.5,0.1,0.2]})
print(df)


plt.scatter(df.X,df.y)
plt.show()

from  sklearn.cluster import KMeans
c=KMeans(n_clusters=2)
c.fit(df)
centroid=c.cluster_centers_
labels=c.labels_
print(centroid)
print(labels)

plt.scatter(df.X,df.y,c=labels)
plt.scatter(centroid[0][0],centroid[0][1])
plt.scatter(centroid[1][0],centroid[1][1])
plt.show()
