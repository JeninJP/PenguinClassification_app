import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("penguins_cleaned.csv")

encode=['sex','island']
for i in encode:
    dum=pd.get_dummies(df[i])
    df=pd.concat([df,dum],axis=1)
    df.drop([i],axis=1,inplace=True)

labelencoder = LabelEncoder()
df['species'] = labelencoder.fit_transform(df['species'])

X=df.drop('species',axis=1)
Y=df['species']

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(X,Y)

import pickle
pickle.dump(clf,open('penguin_clf.pkl','wb'))