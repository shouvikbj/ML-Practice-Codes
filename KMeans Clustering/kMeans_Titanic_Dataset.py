
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

"""
Pclass Passenger Class (1=1st; 2=2nd; 3=3rd)
survival Survival (0=No; 1=Yes)
name Name
sex Sex
age Age
sibsp Number of ssiblings/spouses aboard
parch Number of parents/children aboard
ticket Ticket Number
fare Passenger Fare (British Pound)
cabin Cabin
embarked Port of Embarkation (C=Cherbourg; Q=Queenstown; S=Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
"""

df = pd.read_excel('./KMeans Clustering/titanic.xls')
#print(df.head())
df.drop(['body','name'], 1, inplace=True)
#print(df.head())
##df.convert_objects(convert_numeric=True)  #### 'convert_objects()' is deprecated from pandas v0.18+
#### for the above reason we have to use 'pd.to_numeric' as described below
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
#print(df.head())


#### the below line is not necessary, because strangely somehow ['boat','sex','ticket'] have a huge impact in the result..
##df.drop(['boat','sex','ticket'], 1, inplace=True)

x = np.array(df.drop(['survived'], 1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(x)

correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] ==y[i]:
        correct += 1

print(correct/len(x))

