# DADS6005 Final project realtime analytics of two Cryptocurrency (i.e. BTC and DOGE)

<img width="624" alt="Screenshot 2566-01-11 at 14 55 44" src="https://user-images.githubusercontent.com/80901294/211749515-91287750-e56c-4167-80dd-941f83189f2f.png">

# Introduction

<img width="674" alt="Screenshot 2566-01-11 at 14 34 08" src="https://user-images.githubusercontent.com/80901294/211745323-d3950e17-ba7d-48cb-83ba-66899b940b6c.png">

In this project, we aim to perform real-time analysis of two popular cryptocurrencies: Bitcoin and Dogecoin. Our goal is to predict the future price movements of these currencies by training a machine learning model using the river library's linear regression algorithm. By analyzing historical and realtime data, our model will be able to predict whether the price of a currency will go up (represented as 1) or down (represented as 0). This information can be used by customers to make informed decisions about when to buy or sell these currencies.

# Methods
we initially import free binance api and train bitcoin with linear regression model 
```
import dash
import dash_html_components as html
import dash_core_components as dcc
import requests
import datetime
from dash.dependencies import Input, Output
import numpy as np
from river import neighbors
from river import metrics
import pickle
import sys
from flask import request
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing
from river import optim

#Document of Binance API
#https://binance-docs.github.io/apidocs/spot/en/#rolling-window-price-change-statistics

coin = 'BTCUSDT'
tick_interval = '1m'
key = "https://api.binance.com/api/v3/ticker?symbol="+coin+"&windowSize="+tick_interval

model = (
     preprocessing.StandardScaler() |
     linear_model.LinearRegression(intercept_lr=.1)
)
metric = metrics.MAE()

figure = dict(
    data=[{'x': [], 'y': []}], 
    layout=dict(
        xaxis=dict(range=[]), 
        yaxis=dict(range=[17000, 18000])
        )
    )

app = dash.Dash(__name__, update_title=None)  # remove "Updating..." from title

app.layout = html.Div(
        [
            dcc.Graph(id='graph', figure=figure), 
            dcc.Interval(id="interval",interval=1*60*1000)
        ]
    )

@app.callback(
    Output('graph', 'extendData'), 
    [Input('interval', 'n_intervals')])
    
def update_data(n_intervals):

    print("interval ",n_intervals)

    data = requests.get(key)  
    data = data.json()
    print(data)

    open_price = float(data['openPrice']) #price
    #print(open_price)

    diff_price = float(data['priceChange'])
    print(diff_price)

    closeTime = data['openTime']
    #print(closeTime)

    my_datetime = datetime.datetime.fromtimestamp(closeTime / 1000)  # Apply fromtimestamp function
    print(my_datetime)
    
    keys_to_extract = ['weightedAvgPrice', 'highPrice', 'lowPrice', 'lastPrice', 'volume', 'openPrice', 'quoteVolume', 'openTime', 'closeTime', 'count']
    x = {key: float(data[key]) for key in keys_to_extract}
    print('x : ', x)

    y = float(data['priceChange'])
    print('y : ', y)

    #return y

    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)
    #print('model : ',model)
    print('MAE :', metric)

    return dict(x=[[my_datetime]], y=[[open_price]])
    
if __name__ == '__main__':
    app.run_server()

# save the model to disk
filename = 'updown_prediction_linear.sav'
pickle.dump(model, open(filename, 'wb'))
```

Then we train another model with Dogecoin
```
import dash
import dash_html_components as html
import dash_core_components as dcc
import requests
import datetime
from dash.dependencies import Input, Output
import numpy as np
from river import neighbors
from river import metrics
import pickle
import sys
from flask import request
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing
from river import optim

#Document of Binance API
#https://binance-docs.github.io/apidocs/spot/en/#rolling-window-price-change-statistics

coin = 'DOGEUSDT'
tick_interval = '1m'
key = "https://api.binance.com/api/v3/ticker?symbol="+coin+"&windowSize="+tick_interval

model = (
     preprocessing.StandardScaler() |
     linear_model.LinearRegression(intercept_lr=.1)
)
metric = metrics.MAE()

figure = dict(
    data=[{'x': [], 'y': []}], 
    layout=dict(
        xaxis=dict(range=[]), 
        yaxis=dict(range=[0.075, 0.077])
        )
    )

app = dash.Dash(__name__, update_title=None)  # remove "Updating..." from title

app.layout = html.Div(
        [
            dcc.Graph(id='graph', figure=figure), 
            dcc.Interval(id="interval",interval=1*60*1000)
        ]
    )

@app.callback(
    Output('graph', 'extendData'), 
    [Input('interval', 'n_intervals')])
    
def update_data(n_intervals):

    print("interval ",n_intervals)

    data = requests.get(key)  
    data = data.json()
    print(data)

    open_price = float(data['openPrice']) #price
    #print(open_price)

    diff_price = float(data['priceChange'])
    print(diff_price)

    closeTime = data['openTime']
    #print(closeTime)

    my_datetime = datetime.datetime.fromtimestamp(closeTime / 1000)  # Apply fromtimestamp function
    print(my_datetime)
    
    keys_to_extract = ['weightedAvgPrice', 'highPrice', 'lowPrice', 'lastPrice', 'volume', 'openPrice', 'quoteVolume', 'openTime', 'closeTime', 'count']
    x = {key: float(data[key]) for key in keys_to_extract}
    print('x : ', x)

    y = float(data['priceChange'])
    print('y : ', y)

    #return y

    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    metric.update(y, y_pred)
    #print('model : ',model)
    print('MAE :', metric)

    return dict(x=[[my_datetime]], y=[[open_price]])
    
if __name__ == '__main__':
    app.run_server()

# save the model to disk
filename = 'updown_prediction_doge_linear.sav'
pickle.dump(model, open(filename, 'wb'))
```

We'll be combining two models and training them with historical data and then also be incorporating real-time data to improve our predictions. The goal is to figure out if the price of the crypto is likely to go up or down, which can help people decide when to buy or sell.
```
import dash
from dash import html
from dash import dcc
import requests
import datetime
from dash.dependencies import Input, Output
from river import neighbors
from river import metrics
import pickle
import sys
import numpy as np
from flask import request
from river import datasets
from river import evaluate
from river import linear_model
from river import metrics
from river import preprocessing

#Document of Binance API
#https://binance-docs.github.io/apidocs/spot/en/#rolling-window-price-change-statistics

coin = 'BTCUSDT'
tick_interval = '1m'
key = "https://api.binance.com/api/v3/ticker?symbol="+coin+"&windowSize="+tick_interval

model = (
     preprocessing.StandardScaler() |
     linear_model.LinearRegression(intercept_lr=.1)
)
metric = metrics.MAE()

# load the model from disk
filename = 'updown_prediction_btc_linear.sav'
loaded_model = pickle.load(open(filename, 'rb'))

inf_y = []
op_price = []

figure = dict(
    data=[{'x': [], 'y': []}], 
    layout=dict(
        xaxis=dict(range=[]), 
        yaxis=dict(range=[-10, 10])
        )
    )

coin2 = 'DOGEUSDT'
tick_interval2 = '1m'
key2 = "https://api.binance.com/api/v3/ticker?symbol="+coin2+"&windowSize="+tick_interval2

model2 = (
     preprocessing.StandardScaler() |
     linear_model.LinearRegression(intercept_lr=.1)
)
metric2 = metrics.MAE()

# load the model from disk
filename2 = 'updown_prediction_doge_linear.sav'
loaded_model2 = pickle.load(open(filename2, 'rb'))

inf_y2 = []
op_price2 = []

figure2 = dict(
    data=[{'x': [], 'y': []}], 
    layout=dict(
        xaxis=dict(range=[]), 
        yaxis=dict(range=[-0.0005, 0.0005])
        )
    )

app = dash.Dash(__name__, update_title=None)  # remove "Updating..." from title

app.layout = html.Div(
        [   
            html.H1(children='Bitcoin graph'),
            html.H2(children='This graph show the difference price of Bitcoin from Binance.'), 
            dcc.Graph(id='graph', figure=figure), 
            dcc.Interval(id="interval",interval= 60*1000),
            html.H1(children='Dogecoin graph'),
            html.H2(children='This graph show the difference price of Dogecoin from Binance.'),
            dcc.Graph(id='graph2', figure=figure2), 
            dcc.Interval(id="interval2",interval= 60*1000)
        ]
    )

## chart btc
@app.callback(
    Output('graph', 'extendData'), 
    [Input('interval', 'n_intervals')])

def update_data(n_intervals):

    print("interval ",n_intervals)

    data = requests.get(key)  
    data = data.json()
    #print(data)

    open_price = float(data['openPrice']) #price
    #print(open_price)

    diff_price = float(data['priceChange'])
    print('priceChange', diff_price)

    closeTime = data['openTime']
    #print(closeTime)

    my_datetime = datetime.datetime.fromtimestamp(closeTime / 1000)  # Apply fromtimestamp function
    #print(my_datetime)
    
    keys_to_extract = ['weightedAvgPrice', 'highPrice', 'lowPrice', 'lastPrice', 'volume', 'openPrice', 'quoteVolume', 'openTime', 'closeTime', 'count']
    x = {key: float(data[key]) for key in keys_to_extract}
    #print('x : ', x)

    y = float(data['priceChange'])
    #print('y : ', y)

    #return y

    y_pred = loaded_model.predict_one(x)
    loaded_model.learn_one(x,y)
    #metric.update(y, y_pred)
    #print(y_pred)
    bi_y = np.where(y_pred > 0 , 1, 0) #where 1 is up and 0 is down
    print('Bitcoin prediction-Buy(1) or Sell(0) :', bi_y)

    inf_y.append(bi_y)
    op_price.append(open_price)

    return dict(x = [[my_datetime]], y = [[diff_price]])

## chart doge
@app.callback(
    Output('graph2', 'extendData'), 
    [Input('interval2', 'n_intervals')])

def update_data(n_intervals):

    print("interval ",n_intervals)

    data = requests.get(key2)  
    data = data.json()
    #print(data)

    open_price = float(data['openPrice']) #price
    #print(open_price)

    diff_price = float(data['priceChange'])
    print('priceChange',diff_price)

    closeTime = data['openTime']
    #print(closeTime)

    my_datetime = datetime.datetime.fromtimestamp(closeTime / 1000)  # Apply fromtimestamp function
    #print(my_datetime)
    
    keys_to_extract = ['weightedAvgPrice', 'highPrice', 'lowPrice', 'lastPrice', 'volume', 'openPrice', 'quoteVolume', 'openTime', 'closeTime', 'count']
    x = {key: float(data[key]) for key in keys_to_extract}
    #print('x : ', x)

    y = float(data['priceChange'])
    #print('y : ', y)

    #return y

    y_pred = loaded_model2.predict_one(x)
    loaded_model2.learn_one(x,y)
    #metric.update(y, y_pred)
    #print(y_pred)
    bi_y = np.where(y_pred > 0 , 1, 0) #where 1 is up and 0 is down
    print('Dogecoin prediction-Buy(1) or Sell(0) :', bi_y)

    inf_y2.append(bi_y)
    op_price2.append(open_price)

    return dict(x = [[my_datetime]], y = [[diff_price]])

if __name__ == '__main__':
    app.run_server()

# show list of y and y_pred
print('List of y_pred : ', inf_y)
print('List of open price : ', op_price)

print('List of y_pred : ', inf_y2)
print('List of open price : ', op_price2)

filename = 'updown_prediction_btc_linear2.sav'
pickle.dump(model, open(filename, 'wb'))

filename = 'updown_prediction_doge_linear2.sav'
pickle.dump(model2, open(filename, 'wb'))
```

# Result

**Bitcoin**

<img width="358" alt="Screenshot 2566-01-11 at 14 52 13" src="https://user-images.githubusercontent.com/80901294/211749574-03b1eb0f-65ef-4d81-8973-3f08de73008c.png">

**Dogecoin**

<img width="360" alt="Screenshot 2566-01-11 at 14 52 28" src="https://user-images.githubusercontent.com/80901294/211749621-155fd46e-2f0b-4daa-b4c8-02954ad410ef.png">

As a result of our project, we are able to generate profit from this model, which indicates that Dogecoin has the potential to generate a higher return than Bitcoin. It's worth noting that this experimental result was obtained by training the model for only 2 hours. This suggests that with more training, the model's accuracy and potential for profit generation could be further improved.


![22](https://user-images.githubusercontent.com/80901294/211754923-a1df6875-c30f-46bd-bd37-2a4cf86e890b.jpg)

