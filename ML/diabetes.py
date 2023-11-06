import pandas as pd
import numpy as np
from sklearn import metrics

df=pd.read_csv('diabetes.csv')

df

df.shape

df.isnull().any().value_counts()

df.columns

df_x=df.drop('Outcome',axis=1)
df_y=df['Outcome']

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scaledX=scale.fit_transform(df_x)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test= train_test_split(scaledX, df_y,test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

cs=metrics.confusion_matrix(y_pred,y_test)
print("Confusion Matrix: \n", cs)

ac=metrics.accuracy_score(y_pred,y_test)
print("Accuracy Score: ",ac)

er=1-ac
print("Error Rate: ", er)

p=metrics.precision_score(y_pred,y_test)
print("Precision Score: ", p)

r=metrics.recall_score(y_pred,y_test)
print("Recall: ",r)

cr=metrics.classification_report(y_pred,y_test)
print("Classification Report: ", cr)