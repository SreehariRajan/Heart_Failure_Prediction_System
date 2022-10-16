# Heart_Failure_Prediction

## EDA

Imbalanced data
![image](https://user-images.githubusercontent.com/77159058/196044530-4b6f2a9e-b7d9-4c81-a912-f0bd024d85e5.png)


![image](https://user-images.githubusercontent.com/77159058/196044156-9a70c8f9-a247-4f0a-a069-1c1b8f0caea0.png)

No. of null values:-   
age                         0  
anaemia                     0  
creatinine_phosphokinase    0  
diabetes                    0  
ejection_fraction           0  
high_blood_pressure         0  
platelets                   0  
serum_creatinine            0  
serum_sodium                0  
sex                         0  
smoking                     0  
time                        0  
DEATH_EVENT                 0  
  
dtype: int64  
Shape of the X_train (209, 12)  
Shape of the y_train (209,)  
Shape of the X_test (90, 12)  
Shape of the y_test (90,)  
  
  
## Models


### Logistic regression

-Accuracy score: 0.7888888888888889  
-Precision score: 0.7647058823529411  
-Recall Score: 0.4642857142857143  
-Confusion Matrix: [[58  4][15 13]]  


### Logistic regression with scaling

-Accuracy score: 0.8111111111111111  
-Precision score: 0.7894736842105263  
-Recall Score: 0.5357142857142857  
-Confusion Matrix: [[58  4][13 15]]  


### Support vector classifier

-SVC(C=10, gamma=0.0001)  
-Accuracy score: 0.6777777777777778  
-Precision score: 0.07142857142857142  
-Recall Score: 0.4  
-Confusion Matrix: [[59 26][ 3  2]]  


### Decision tree classifier

-Accuracy score: 0.7555555555555555
-Precision score: 0.625
-Recall Score: 0.5357142857142857
-Confusion Matrix: [[53  9][13 15]]


### Random forest classifier

-Accuracy score: 0.8666666666666667
-Precision score: 0.9
-Recall Score: 0.6428571428571429
-Confusion Matrix: [[60  2][10 18]]


### XGBoost classifier

-Accuracy score: 0.8555555555555555
-Precision score: 0.8
-Recall Score: 0.7142857142857143
-Confusion Matrix: [[57  5][ 8 20]]


### Gradient Boosting classifier

-Accuracy score: 0.8555555555555555
-Precision score: 0.8571428571428571
-Recall Score: 0.6428571428571429
-Confusion Matrix: [[59  3][10 18]]
