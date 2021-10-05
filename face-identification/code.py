import pandas as pd
import numpy as np

iden=pd.read_csv("/test.csv")
iden

x=iden.iloc[:,:-1].values
x

y=iden.iloc[:,-1].values
y

x=pd.DataFrame(x)
x

y=pd.DataFrame(y)
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()

model.fit(x_train, y_train)

model.predict(x_test)

model.score(x_test, y_test)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model1=DecisionTreeClassifier()

model1.fit(x_train, y_train)

model1.predict(x_test)

model1.score(x_test, y_test)

#KNeghborsClassifier
from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier()

model2.fit(x_train, y_train)

model2.predict(x_test)

model2.score(x_test, y_test)
