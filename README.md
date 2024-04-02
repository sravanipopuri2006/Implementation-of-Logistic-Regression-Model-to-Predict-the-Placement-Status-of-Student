# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.
```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: POPURI SRAVANI
RegisterNumber:  212223240117
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1.head()

x=data1.iloc[:,:-1]
print(x)

y=data1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import accuracy_score
confusion=(y_test,y_pred)
print(confusion)

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## DATASET
![Screenshot 2024-04-02 131700](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/99825a4b-80c6-441c-acf4-024df78a0e84)
## DATASET AFTER DROPPING THE SALARY COLUMN
![Screenshot 2024-04-02 131708](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/8e26a192-801d-4c79-a14a-0ac3dae03761)
## CHECKING IF NULL VALUES ARE PRESENT
![Screenshot 2024-04-02 131714](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/3679585e-943e-41b2-9da4-ae3097c2750f)
## CHECKING IF DUPLICATE VALUES ARE PRESENT
![Screenshot 2024-04-02 131726](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/6179756a-ea7a-49fb-8f64-17c1c6b6aecc)
## DATASET AFTER ENCODING
![Screenshot 2024-04-02 131736](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/3e7155e0-2ae1-416b-b2ab-71ae6cf09637)
## X-VALUES
![Screenshot 2024-04-02 131750](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/65e24c06-eb74-4dd1-8be4-01081339e657)
## Y-VALUES
![Screenshot 2024-04-02 131804](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/f10a98a8-6ede-4652-ad61-2cbc9f0f5aa7)
## Y_PRED VALUES
![Screenshot 2024-04-02 131815](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/14806a6d-ed8a-4610-98a8-cf386e517579)
## ACCURACY
![Screenshot 2024-04-02 131826](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/499a2d67-db3a-47fb-a061-1ed4458b2ce6)
## CONFUSION MATRIX
![Screenshot 2024-04-02 131851](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/7eab1ae1-5947-4cda-8834-b27e386468e0)
## CLASSIFICATION_REPORT
![Screenshot 2024-04-02 131903](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/e39f70d2-0e1a-42f0-88e6-33365a86597a)
## lr.predict
![Screenshot 2024-04-02 131913](https://github.com/sravanipopuri2006/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139778301/a0997e99-8cf5-48d7-93f3-a202dff37956)















## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
