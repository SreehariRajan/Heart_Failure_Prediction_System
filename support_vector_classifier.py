import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV 


data = pd.read_csv("dataset/archive/heart_failure_clinical_records_dataset.csv")
data.head()
print("Shape of data",data.shape)
data.info()
print("Describing the data:-")
data.describe()
print("No. of null values:-")
data.isnull().sum()

# explore dataset  
live_len = len(data["DEATH_EVENT"][data.DEATH_EVENT==0])
death_len = len(data["DEATH_EVENT"][data.DEATH_EVENT==1])
len_arr = np.array([live_len,death_len])
plt.pie(len_arr,labels=["LIVING","DIED"],explode=[0.2,0.0],shadow=True)
plt.show()

corr = data.corr()
plt.subplots(figsize=(15,15))
sns.heatmap(corr,annot=True)

# dataset development
X = data.drop('DEATH_EVENT',axis=1)
y=data["DEATH_EVENT"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print("Shape of the X_train",X_train.shape)
print("Shape of the y_train",y_train.shape)
print("Shape of the X_test",X_test.shape)
print("Shape of the y_test",y_test.shape)


# Feature Engineering

def add_interactions(X):
    features = X.columns
    m=len(features)

    X_int = X.copy(deep=True)

    for i in range(m):

        feature_i_name = features[i]
        feature_i_data = X[feature_i_name]

        for j in range(i+1,m):
            feature_j_name = features[j]
            feature_j_data = X[feature_j_name]
            feature_i_j_name = feature_i_name+" X "+feature_j_name
            X_int[feature_i_j_name] = feature_i_data*feature_j_data
    
    return X_int

x_train_mod = add_interactions(X_train)
x_test_mod = add_interactions(X_test)

# model

def eval(y_test,y_pred):
    print("Accuracy score:",accuracy_score(y_test,y_pred))
    print("Precision score:",precision_score(y_test,y_pred))
    print("Recall Score:",recall_score(y_test,y_pred))
    print("Confusion Matrix:",confusion_matrix(y_test,y_pred))



param_grid = {'C':[0.1,1,10,100,1000],
            'gamma':[1,0.1,0.01,0.001,0.0001],
            'kernel':['rbf']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)

# best hyper parameters
print(grid.best_estimator_)

svc = SVC(C=10,gamma=0.0001)
svc.fit(X_train,y_train)
y_pred_3 = svc.predict(X_test)

eval(y_pred_3,y_test)

