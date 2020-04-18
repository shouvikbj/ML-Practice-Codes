import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

quandl.ApiConfig.api_key = 'RpKUdarDmRD23J6yBECW'
quandl.ApiConfig.api_version = '2015-04-09'

df = quandl.get('WIKI/GOOGL')

#print(df.head())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#print(df)

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
#print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1 * len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

#print(df.head())
#print(df.tail())

##x = np.array(df.drop(['label'],1))
##y = np.array(df['label'])
##x = preprocessing.scale(x)
#print(len(x),len(y))

x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

##clf = LinearRegression(n_jobs=20)
##clf2 = svm.SVR()
##clf.fit(x_train, y_train)
##with open('./Google Stock ML/Linear Regression/linearregression.pickle','wb') as f:
##    pickle.dump(clf, f)

pickle_in = open('./Google Stock ML/Linear Regression/linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)
##clf2.fit(x_train, y_train)
##accuracy2 = clf2.score(x_test, y_test)
#print('Linear Regression = >')
#print(round((accuracy * 100),2), '%')
#print('svm.SVR = >')
#print(round((accuracy2 * 100),2), '%')
forecast_set = clf.predict(x_lately)
#print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


