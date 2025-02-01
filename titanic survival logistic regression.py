import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import seaborn as sns
path=r"C:\Users\ADMIN\Downloads\tested.csv"
print(os.path.exists(path))
data=pd.read_csv(path)
# print(data.head())
# print (data.shape)
# print(data.info())
# print(data.isnull().sum())
data['Age']=data['Age'].fillna(data['Age'].mean())
data['Cabin']=data['Cabin'].fillna(data['Cabin'].mode()[0])
data['Fare']=data['Fare'].fillna(data['Fare'].mean())
data.dropna(inplace=True)
print(data.isnull().sum())
print(data.info())
data['Sex']=data['Sex'].replace({'male':0,'female':1})
print(data['Sex'].value_counts())
data=data.drop(['Cabin','Name','Ticket','Embarked'], axis=1 )
print(data)
x=data.drop('Survived',axis=1)
y=data['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape)
print(x_test.shape)
scale=StandardScaler()
x_train_scaled=scale.fit_transform(x_train)
x_test_scaled=scale.transform(x_test)

model=LogisticRegression(max_iter=2000,C=1000000)
model.fit(x_train_scaled,y_train)
prediction=model.predict(x_test_scaled)
print("lr prediction=",metrics.accuracy_score(y_test,prediction))

# print(data.shape)
# plt.figure(figsize=(10,10))
# sns.set(font_scale=1)
# plt.plot(5,3,1)
# sns.histplot(x='Sex',hue='Age' ,data=data)
# plt.title("gender vs age titanic")
# plt.tight_layout()
# plt.show()
