from django.shortcuts import render, redirect
from django.urls import resolve
import numpy as np
import pandas as pd
import math


def round_down(n, decimals):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def home(request, symbol):
    if resolve(request.path_info).url_name == 'home':
        template = 'mysite/home.html'
    else:
        template = 'mysite/markets.html'
    data = np.array(pd.read_csv('mysite/static/mysite/data.csv'))
    print(data[0][0])
    for i in range(len(data)):
        if data[i][0] == symbol:
            prediction = float(data[i][1])
            mse = float(data[i][2])
            prevdata = data[i][3]
            prev_data = data[i][3].split(', ')
            break

    diff = prediction - float(prev_data[-1])
    percent_diff = diff / float(prev_data[-1]) * 100

    if mse >= 0.01:
        rating = 'Very poor'
    elif mse >= 0.001:
        rating = 'Poor'
    elif mse >= 0.0001:
        rating = 'Fair'
    elif mse >= 0.00001:
        rating = 'Good'
    else:
        rating = 'Very good'

    print(prediction, diff, percent_diff, mse, rating, prevdata)
    if diff < 0:
        diff = 'Price Change: -$' + str(diff)
        percent_diff = 'Percent Change: ' + str(percent_diff) + '%'
    else:
        diff = 'Price Change: $' + str(diff)
        percent_diff = 'Percent Change: ' + str(percent_diff) + '%'
    return render(request, template, {'symbol': 'Current Symbol: ' + symbol + ' |',
                                                   'prediction': ' Predicted Closing Price: $' + str(prediction),
                                                   'diff': diff + ' |',
                                                   'percent_diff': ' ' + percent_diff,
                                                   'prevdata': 'Last 7 Closing Prices (present-): ' + prevdata,
                                                   'mse': 'Mean Squared Error (Model Accuracy): ' + str(mse),
                                                   'rating': 'Model Accuracy Rating: ' + rating})


def about(request):
    plot = 'plot.PNG'
    return render(request, 'mysite/about.html', {'plot': plot})


def contact(request):
    return render(request, 'mysite/contact.html')


def markets(request, symbol):
    return home(request, symbol)


def landing_redirect(request):
    return redirect("home/AAPL")


def custom_page_not_found_view(request, exception):
    return render(request, "mysite/404.html", {})


def custom_error_view(request, exception=None):
    return render(request, "mysite/500.html", {})


def custom_permission_denied_view(request, exception=None):
    return render(request, "mysite/403.html", {})


def custom_bad_request_view(request, exception=None):
    return render(request, "mysite/400.html", {})
