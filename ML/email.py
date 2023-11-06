import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

df=pd.read_csv("emails.csv")

df.head()

df.isnull().sum()

X=df.iloc[:,1:3001]
X

Y=df.iloc[:,-1].values
Y

train_x , test_x, train_y, test_y= train_test_split(X,Y,test_size=0.25)

svc=SVC(C=1.0,kernel='rbf',gamma='auto')
svc.fit(train_x,train_y)
y_pred2=svc.predict(test_x)


print("Accuracy score of SVC :", accuracy_score(y_pred2,test_y))

X_train,X_test, y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train,y_train)

print(knn.predict(X_train))

print(knn.score(X_test, y_test))

val=df['Prediction'].value_counts()
val.plot(kind='bar', color='gray')
plt.xlabel('')
plt.ylabel('Quantities')
plt.xticks([0,1],['not spam','spam'])
plt.show()