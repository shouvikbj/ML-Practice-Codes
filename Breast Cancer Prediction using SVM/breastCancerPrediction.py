import numpy as np
import pandas as pd
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split


df = pd.read_csv('./Breast Cancer Prediction using SVM/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = svm.SVC()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)

print('Accuracy => ', accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[10,5,8,2,1,10,9,3,4]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)

print(prediction)


