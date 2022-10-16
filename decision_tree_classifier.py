import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV  


data = pd.read_csv("dataset/archive/heart_failure_clinical_records_dataset.csv")
data.head()
print("Shape of data",data.shape)
data.info()
print("Describing the data:-",data.describe())

print("No. of null values:-",data.isnull().sum())

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


def randomized_search(params,runs=20,clf=DecisionTreeClassifier(random_state=2)):
    rand_clf = RandomizedSearchCV(clf,params,n_iter=runs,cv=5,n_jobs=-1,random_state=2)
    rand_clf.fit(X_train,y_train)
    best_model = rand_clf.best_estimator_
    best_score = rand_clf.best_score_

    print("Training score: {:.3f}".format(best_score))    
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    print('Test score: {:.3f}'.format(accuracy))

    return best_model

print(randomized_search(params={'criterion':['entropy', 'gini'],
                              'splitter':['random', 'best'],
                          'min_weight_fraction_leaf':[0.0, 0.0025, 0.005, 0.0075, 0.01],
                          'min_samples_split':[2, 3, 4, 5, 6, 8, 10],
                          'min_samples_leaf':[1, 0.01, 0.02, 0.03, 0.04],
                          'min_impurity_decrease':[0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
                          'max_leaf_nodes':[10, 15, 20, 25, 30, 35, 40, 45, 50, None],
                          'max_features':['auto', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                          'max_depth':[None, 2,4,6,8],
                          'min_weight_fraction_leaf':[0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05]
                         }))

ds_clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, max_features=0.75,
                       max_leaf_nodes=25, min_impurity_decrease=0.0005,
                       min_samples_split=5, min_weight_fraction_leaf=0.0075,
                       random_state=2)
ds_clf.fit(X_train,y_train)
pred=ds_clf.predict(X_test)

eval(y_test,pred)
