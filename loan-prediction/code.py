import pandas as pd
import numpy as np
import statistics

loan_train=pd.read_csv("/loan-train.csv")
loan_train

loan_test=pd.read_csv("/loan-test.csv")
loan_test

loan_train.isna().sum()

loan_train["Gender"].fillna(statistics.mode(loan_train["Gender"]), inplace= True)
loan_train["Married"].fillna(statistics.mode(loan_train["Married"]), inplace= True)
loan_train["Dependents"].fillna(statistics.mode(loan_train["Dependents"]), inplace= True)
loan_train["Self_Employed"].fillna(statistics.mode(loan_train["Self_Employed"]), inplace= True)
loan_train["ApplicantIncome"].fillna(loan_train["ApplicantIncome"].mean(), inplace = True)
loan_train["CoapplicantIncome"].fillna(loan_train["CoapplicantIncome"].mean(), inplace = True)
loan_train["LoanAmount"].fillna(loan_train["LoanAmount"].mean(), inplace = True)
loan_train["Loan_Amount_Term"].fillna(statistics.mode(loan_train["Loan_Amount_Term"]), inplace= True)
loan_train["Credit_History"].fillna(statistics.mode(loan_train["Credit_History"]), inplace= True)
loan_train["Property_Area"].fillna(statistics.mode(loan_train["Property_Area"]), inplace= True)
loan_train

loan_test.isna().sum()

loan_test["Gender"].fillna(statistics.mode(loan_test["Gender"]), inplace= True)
loan_test["Married"].fillna(statistics.mode(loan_test["Married"]), inplace= True)
loan_test["Dependents"].fillna(statistics.mode(loan_test["Dependents"]), inplace= True)
loan_test["Self_Employed"].fillna(statistics.mode(loan_test["Self_Employed"]), inplace= True)
loan_test["ApplicantIncome"].fillna(loan_test["ApplicantIncome"].mean(), inplace = True)
loan_test["CoapplicantIncome"].fillna(loan_test["CoapplicantIncome"].mean(), inplace = True)
loan_test["LoanAmount"].fillna(loan_test["LoanAmount"].mean(), inplace = True)
loan_test["Loan_Amount_Term"].fillna(statistics.mode(loan_test["Loan_Amount_Term"]), inplace= True)
loan_test["Credit_History"].fillna(statistics.mode(loan_test["Credit_History"]), inplace= True)
loan_test["Property_Area"].fillna(statistics.mode(loan_test["Property_Area"]), inplace= True)
loan_test

x=loan_train.drop(["Loan_Status","Loan_ID"],axis=1)
x

y=loan_train["Loan_Status"]
y

x1=loan_test.drop("Loan_ID",axis=1)
x1

x=pd.DataFrame(x)
x

y=pd.DataFrame(y)
y

x1=pd.DataFrame(x1)
x1

x=pd.get_dummies(x)
x

x1=pd.get_dummies(x1)
x1

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()

model.fit(x, y)

model.predict(x1)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier  
model1=DecisionTreeClassifier()

model.fit(x, y)

model.predict(x1)

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier()

model.fit(x, y)

model.predict(x1)
