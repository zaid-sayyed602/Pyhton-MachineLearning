import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
#-----------Label Encoder-------------------
df=pd.read_csv("cardetails.csv")
print(df.info())
print(df.head())
print(np.array(df['owner']))
le=LabelEncoder()
print(df['owner'])
#df['owner']=le.fit_transform(list(df['owner']))
print(df['owner'])

#----------OneHotEncoder--------------------
oe=OneHotEncoder()
a=np.array(df["owner"].values.tolist()).reshape(-1,1)
oe.fit(a)
print(oe.transform(a).toarray())
