import numpy as np
import yfinance as yf
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
import os
# Lets save as csv and then upload to view and change to numpy. store all data in 1 file

symbols_file = pd.read_csv('static/mysite/nasdaq-listed-symbols_csv.csv')
symbols = symbols_file.pop('Symbol')
company_names = symbols_file.pop('Company Name')
data = []
preddata = []
# symbol_data = []
# prediction_data = []
count = 1

for symbol in range(len(symbols)):
    predict = []
    to_remove = []

    try:
        data = yf.download(tickers=symbols[symbol], period='max', interval='1d', group_by='ticker', auto_adjust=True,
                           prepost=True, proxy=None)
        print(symbols[symbol], count)
        OPEN, high, low, close, volume = data.pop('Open'), data.pop('High'), data.pop('Low'), data.pop(
            'Close'), data.pop('Volume')
        y = close
        y = y[1:]

        for i in range(len(close)):
            if np.isnan(close[i]):
                to_remove.append(i)
        for i in range(len(to_remove)):
            OPEN = OPEN[:to_remove[i] - i] + OPEN[to_remove[i] - i + 1:]
            high = high[:to_remove[i] - i] + high[to_remove[i] - i + 1:]
            low = low[:to_remove[i] - i] + low[to_remove[i] - i + 1:]
            close = close[:to_remove[i] - i] + close[to_remove[i] - i + 1:]
            volume = volume[:to_remove[i] - i] + volume[to_remove[i] - i + 1:]
            y = y[:to_remove[i] - i] + y[to_remove[i] - i + 1:]

        OPEN = (OPEN - min(OPEN)) / (max(OPEN) - min(OPEN))
        high = (high - min(high)) / (max(high) - min(high))
        low = (low - min(low)) / (max(low) - min(low))
        close = (close - min(close)) / (max(close) - min(close))
        volume = (volume - min(volume)) / (max(volume) - min(volume))

        for i in range(len(close)):
            predict.append([OPEN[i], high[i], low[i], close[i], volume[i]])

        predict, train_labels = np.array(predict), np.array(y)
        mx, mn = max(train_labels), min(train_labels)
        tl = train_labels
        train_labels = (train_labels - min(train_labels)) / (max(train_labels) - min(train_labels))
        train_data = predict[:-1]
        train_data = train_data.reshape(-1, 5, 1)

        model = Sequential()
        model.add(LSTM(256, activation='sigmoid', kernel_initializer='he_normal', input_shape=(train_data.shape[1:]),
                       return_sequences=True))
        model.add(LSTM(64, activation='sigmoid', kernel_initializer='he_normal', return_sequences=True))
        model.add(LSTM(64, activation='sigmoid', kernel_initializer='he_normal'))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        model.fit(train_data, train_labels, epochs=10, batch_size=256, verbose=0)
        mse, mae = model.evaluate(train_data, train_labels, verbose=0)
        print('MSE: %.3f,  MAE: %.3f' % (mse, mae))

        predictions = model.predict(predict.reshape(-1, 5, 1))
        prediction = float(predictions[-1] * (mx - mn) + mn)
        print('Previous:', tl[-1], 'Prediction:', prediction)
        preddata.append([symbols[symbol], prediction, mse, np.array2string(np.flipud(tl[-7:]), separator=', ')[1:-1]])

        df = pd.DataFrame(np.array(preddata))
        for i in os.listdir('static/mysite/'):
            if i == 'data.csv':
                os.unlink('static/mysite/data.csv')
        df.to_csv('static/mysite/data.csv', index=False)
        # change to csv
        # data_file = open('static/mysite/prediction_data.txt', 'w')
        # for i in prediction_data:
        #     data_file.write(str(i))
        #     data_file.write(' ')
        # data_file.close()
        #
        # data_file = open('static/mysite/symbol_data.txt', 'w')
        # for i in symbol_data:
        #     data_file.write(str(i))
        #     data_file.write(' ')
        # data_file.close()
        # print('Predictions saved')

    except ValueError:
        print('Err~~')
        count -= 1

    count += 1

########################################################################################################################
# import numpy as np
# import yfinance as yf
# import pandas as pd
# from matplotlib.figure import Figure
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, LSTM
# import os
# import shutil
#
# symbols_file = pd.read_csv('static/mysite/nasdaq-listed-symbols_csv.csv')
# symbols = symbols_file.pop('Symbol')
# company_names = symbols_file.pop('Company Name')
#
# for file in os.listdir('static/mysite/saved_model_weights/'):
#     file_path = os.path.join('static/mysite/saved_model_weights/', file)
#     if os.path.isfile(file_path):
#         os.unlink(file_path)
#     else:
#         shutil.rmtree(file_path)
#
# for file in os.listdir('static/mysite/plots/'):
#     file_path = os.path.join('static/mysite/plots/', file)
#     os.unlink(file_path)
#
# for i in range(len(symbols)):
#     to_remove = []
#     count = 0
#     try:
#         data = yf.download(tickers=symbols[i], period='max', interval='1d', group_by='ticker', auto_adjust=True,
#                            prepost=True, proxy=None)
#         print(symbols[i], i)
#
#         train_data = np.array(data.pop('Open'))
#         train_labels = np.array(data.pop('Close'))
#         for _ in range(len(train_data)):
#             if np.isnan(train_data[_]):
#                 to_remove.append(_)
#
#         for _ in to_remove:
#             train_data = np.delete(train_data, _ - count)
#             count += 1
#         to_remove.clear()
#         count = 0
#
#         for _ in range(len(train_labels)):
#             if np.isnan(train_labels[_]):
#                 to_remove.append(_)
#
#         for _ in to_remove:
#             train_labels = np.delete(train_labels, _ - count)
#             count += 1
#
#         test = train_data
#         graph = train_data
#         close = train_labels
#         test = test.reshape(-1, 1, 1)
#         graph = graph.reshape(-1, 1, 1)
#         mx = max(train_data)
#         mn = min(train_data)
#         graph = (graph - mn) / (mx - mn)
#         train_data, train_labels = (train_data - mn) / (mx - mn), (train_labels - mn) / (mx - mn)
#         train_data = train_data.reshape(-1, 1, 1)
#
#         model = Sequential()
#         model.add(LSTM(256, activation='sigmoid', kernel_initializer='he_normal', input_shape=(train_data.shape[1:])))
#         model.add(Dense(128, activation='sigmoid', kernel_initializer='he_normal'))
#         model.add(Dense(64, activation='sigmoid', kernel_initializer='he_normal'))
#         model.add(Dense(1, activation='linear'))
#
#         model.compile(optimizer='adam', loss='mse', metrics=['mse'])
#         model.fit(train_data, train_labels, epochs=18, batch_size=256)
#         mse, mae = model.evaluate(train_data, train_labels, verbose=0)
#         print('MSE: %.3f,  MAE: %.3f' % (mse, mae))
#
#         predictions = model.predict(graph)
#         for _ in range(len(predictions)):
#             predictions[_] = predictions[_] * (mx - mn) + mn
#
#         fig = Figure()
#         ax = fig.subplots()
#         ax.plot(predictions, linewidth=0.6, label='Predicted Closing Price')
#         ax.plot(close, linewidth=0.6, label='Actual Closing Price')
#         ax.set_title(symbols[i] + ' (' + company_names[i] + ')' + ' Results From Model Training')
#         ax.set_xlabel('Days since IPO')
#         ax.set_ylabel('Price ($)')
#         ax.legend()
#         fig.savefig('static/mysite/plots/' + symbols[i] + '.PNG', dpi=300)
#
#         model.save('static/mysite/saved_model_weights/' + symbols[i] + '_saved_model.h5')
#         data_file = open(symbols[i] + ' data.txt', 'w')
#         for _ in test:
#             data_file.write(str(_[0][0]))
#             data_file.write(' ')
#         data_file.close()
#         os.rename(symbols[i] + ' data.txt', 'static/mysite/saved_model_weights/' + symbols[i] + ' data.txt')
#         print('Model saved')
#
#     except:
#         pass
