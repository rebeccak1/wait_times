import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import plotly.graph_objs as go

import numpy as np
import time

import datetime

import pandas as pd

import itertools
	
if __name__ == '__main__':
    dateparse = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv('newdata-11-6.txt', parse_dates=['Date'], index_col=False,date_parser=dateparse, delimiter=',')
    waits = df['Waits']
    date = df['Date']

    times = pd.date_range('07:00', '23:45', freq='15min').time

    #find number of unique days
    #print sorted(set(map(lambda d: d.strftime('%m-%d'), date)))
    unique_days = len(set(map(lambda d: d.strftime('%m-%d'), date)))
    unique_days = unique_days/7.

    z=list(map(lambda d: map(lambda t: sum(x[1]/unique_days for x in filter(lambda f: f[0].strftime("%H:%M") == t.strftime("%H:%M"), filter(lambda day: day[0].strftime("%a") == d, zip(date, waits)))), times), ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']))

    print map(lambda t: filter(lambda f: f[0].strftime("%H:%M") == t.strftime("%H:%M"), filter(lambda day: day[0].strftime("%a") == 'Mon', zip(date, waits))), times)
    #print reduce(lambda wait1, wait2: wait1+wait2,filter(lambda day: day[0].strftime("%a") == 'Mon', zip(date, waits)))

    data = [
	go.Heatmap(
	    z=z,
	    x=times,
	    y=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
	)
    ]

    layout = go.Layout(
	title='Average wait per day and time',
	xaxis = dict(ticks='',nticks=36),
	yaxis = dict(ticks='' )
    )
    fig = go.Figure(data=data, layout=layout)

    unique_url = py.plot(fig, filename='sevendwarfs=heatmap')
