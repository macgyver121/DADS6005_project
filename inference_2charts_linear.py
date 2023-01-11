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