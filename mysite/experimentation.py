import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM


predict = []
to_remove = []
MSEs = []
directions = []
preds = []
pp = []
tp = []
iters = 300

data = yf.download(tickers='TSLA', period='max', interval='1d', group_by='ticker', auto_adjust=True,
                   prepost=True, proxy=None)

open, high, low, close, volume = data.pop('Open'), data.pop('High'), data.pop('Low'), data.pop('Close'), data.pop('Volume')
y = close
y = y[1:]

for i in range(len(close)):
    if np.isnan(close[i]):
        to_remove.append(i)
for i in range(len(to_remove)):
    open = open[:to_remove[i] - i] + open[to_remove[i] - i + 1:]
    high = high[:to_remove[i] - i] + high[to_remove[i] - i + 1:]
    low = low[:to_remove[i] - i] + low[to_remove[i] - i + 1:]
    close = close[:to_remove[i] - i] + close[to_remove[i] - i + 1:]
    volume = volume[:to_remove[i] - i] + volume[to_remove[i] - i + 1:]
    y = y[:to_remove[i] - i] + y[to_remove[i] - i + 1:]

open = (open - min(open)) / (max(open) - min(open))
high = (high - min(high)) / (max(high) - min(high))
low = (low - min(low)) / (max(low) - min(low))
close = (close - min(close)) / (max(close) - min(close))
volume = (volume - min(volume)) / (max(volume) - min(volume))

for i in range(len(close)):
    predict.append([open[i], high[i], low[i], close[i], volume[i]])

predict, train_labels = np.array(predict), np.array(y)
mx, mn = max(train_labels), min(train_labels)
tl = train_labels
train_labels = (train_labels - min(train_labels)) / (max(train_labels) - min(train_labels))
train_data = predict[:-1]
otd, otl = train_data, train_labels

for i in range(iters - 1, -1, -1):
    print(i + 1, '↓')
    graph = predict[:-i - 1]
    target = tl[-i - 1]
    graph = graph.reshape(-1, 5, 1)
    train_data, train_labels = otd[:-i - 1], otl[:-i - 1]
    train_data = train_data.reshape(-1, 5, 1)

    model = Sequential()
    model.add(LSTM(512, activation='sigmoid', kernel_initializer='he_normal', input_shape=(train_data.shape[1:]), return_sequences=True))
    model.add(LSTM(128, activation='sigmoid', kernel_initializer='he_normal', return_sequences=True))
    model.add(LSTM(128, activation='sigmoid', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.fit(train_data, train_labels, epochs=15, batch_size=128, verbose=1)
    mse, mae = model.evaluate(train_data, train_labels, verbose=0)

    predictions = model.predict(graph)
    prediction = float(predictions[-1]) * (mx - mn) + mn
    print('Previous:', tl[-i - 2])
    print(prediction, target)
    print((prediction - (tl[-i - 2])) / tl[-i - 2] * 100, (target - tl[-i - 2]) / tl[-i - 2] * 100)
    tp.append((target - tl[-i - 2]) / tl[-i - 2] * 100)
    pp.append((prediction - tl[-i - 2]) / tl[-i - 2] * 100)
    preds.append(prediction)
    MSEs.append(abs(((target - tl[-i - 2]) / tl[-i - 2] * 100) - ((prediction - tl[-i - 2]) / tl[-i - 2] * 100)))
    if (((target - tl[-i - 2]) / tl[-i - 2] * 100) >= 0 and ((prediction - tl[-i - 2]) / tl[-i - 2] * 100) >= 0) or (((target - tl[-i - 2]) / tl[-i - 2] * 100) <= 0 and ((prediction - tl[-i - 2]) / tl[-i - 2] * 100) <= 0):
        directions.append(1)
    else:
        directions.append(0)

print('MSEs:', MSEs)
print('Mean MSEs:', np.mean(MSEs))
print('Mean directional similarity:', np.mean(directions))

fig, axs = plt.subplots(3)
axs[0].plot(preds, color='blue')
axs[0].plot(tl[-iters:], color='orange')
axs[0].set_title('Predicted Price x Actual')
axs[0].axes.xaxis.set_visible(False)
axs[1].plot(pp, color='blue')
axs[1].plot(tp, color='orange')
axs[1].set_title('Predicted Percent Change x Actual Percent Change')
axs[1].axes.xaxis.set_visible(False)
axs[2].plot(MSEs, color='blue')
axs[2].set_title('Predicted x Actual Percentage Difference')
plt.savefig('plot')
plt.show()
