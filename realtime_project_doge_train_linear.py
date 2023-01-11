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