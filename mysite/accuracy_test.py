import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

symbols_file = pd.read_csv('static/mysite/nasdaq-listed-symbols_csv.csv')
symbols = symbols_file.pop('Symbol')
company_names = symbols_file.pop('Company Name')

to_remove = []
count = 0
MSEs = []
directions = []
preds = []
iters = 100
data = yf.download(tickers='GOOG', period='max', interval='1d', group_by='ticker', auto_adjust=True,
                   prepost=True, proxy=None)

print(data.head())
train_data = np.array(data.pop('Open'))
train_labels = np.array(data.pop('Close'))
for _ in range(len(train_data)):
    if np.isnan(train_data[_]):
        to_remove.append(_)

for _ in to_remove:
    train_data = np.delete(train_data, _ - count)
    count += 1
to_remove.clear()
count = 0

for _ in range(len(train_labels)):
    if np.isnan(train_labels[_]):
        to_remove.append(_)

for _ in to_remove:
    train_labels = np.delete(train_labels, _ - count)
    count += 1

td, tl = train_data, train_labels
for i in range(iters):
    print(i)
    train_data, train_labels = td, tl
    graph = train_data[:len(train_data) - 1 - i]
    target = train_labels[-i - 1]
    graph = graph.reshape(-1, 1, 1)
    train_data, train_labels = train_data[:-i - 1], train_labels[:-i - 1]
    mx = max(train_data)
    mn = min(train_data)
    graph = (graph - min(graph)) / (max(graph) - min(graph))
    train_data, train_labels = (train_data - mn) / (mx - mn), (train_labels - mn) / (mx - mn)
    train_data = train_data.reshape(-1, 1, 1)

    model = Sequential()
    model.add(LSTM(256, activation='sigmoid', kernel_initializer='he_normal', input_shape=(train_data.shape[1:])))
    model.add(Dense(128, activation='sigmoid', kernel_initializer='he_normal'))
    model.add(Dense(64, activation='sigmoid', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.fit(train_data, train_labels, epochs=20, batch_size=64)
    mse, mae = model.evaluate(train_data, train_labels, verbose=0)

    predictions = model.predict(graph)
    prediction = predictions[-1] * (mx - mn) + mn
    preds.append(float(prediction))
    MSEs.append(abs(((td[-i - 2] - target) / target * 100) - (float((td[-i - 2] - prediction) / prediction) * 100)))
    print(abs(((td[-i - 2] - target) / target * 100) - (float((td[-i - 2] - prediction) / prediction) * 100)))
    if (((td[-i - 2] - target) / target * 100) > 0 and (float((td[-i - 2] - prediction) / prediction) * 100) > 0) or (((td[-i - 2] - target) / target * 100) < 0 and (float((td[-i - 2] - prediction) / prediction) * 100) < 0):
        directions.append(1)
    else:
        directions.append(0)

MSEs.reverse()
print('MSEs:', MSEs)
print('Mean MSEs:', np.mean(MSEs))
print('Mean directional similarity:', np.mean(directions))

plt.plot(preds, color='blue')
plt.plot(td[-iters:], color='orange')
plt.legend()
plt.show()
