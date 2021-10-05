import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

boston=load_boston()
boston

boston_df=pd.DataFrame(boston['data'],columns=boston['feature_names'])
boston_df

boston_df["y"]=pd.Series(boston['target'])
boston_df

x=boston_df.iloc[:,:-1].values
x

y=boston_df.iloc[:,-1].values
y

x=pd.DataFrame(x)
x

y=pd.DataFrame(y)
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()

model.fit(x_train, y_train)

model.predict(x_test)

model.score(x_test, y_test)
