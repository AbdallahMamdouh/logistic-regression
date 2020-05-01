from LogisticRegression import LogisticRegression,featureScale
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression as LR

#loading data
data = pd.read_csv('data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#scaling X
X=featureScale(X)

#splitting data to test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#initializing model
model=LogisticRegression()

#training the model
J,theta=model.train(X_train,y_train)

#predicting using X_test data
y_pred=model.predict(X_test)
y_pred=np.round(y_pred)

#visualizing confusion matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#visualizing cost function per iteration
plt.plot(J)
plt.show()

#visualizing mean square error and root mean square error
MSE=J[-1]
RMSE=np.sqrt(MSE)
print("mean square error: ",MSE)
print("root mean square error: ",RMSE)