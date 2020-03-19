# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:37:40 2019

@author: World peace
"""

import numpy as np
import pandas as pd
import datetime
import pytz

## read dataset
#order_dataset = pd.read_csv('order_dataset.csv', encoding='ISO-8859-1')

## data normalize
### convert timestamp
def data_normalize(order_dataset):
    hour = []
    day = []
    #date = []
    month = []
    weekday = []
    minute = []
    lat = []
    long = []
    pst = pytz.timezone('America/Vancouver')
    for i in range(len(order_dataset)):
        ## lat long
        lat_, lng_ = str(order_dataset['latlng_from'][i]).split(',')
        
        ## time
        d = datetime.datetime.utcfromtimestamp(np.array(order_dataset['finish_time'][i]))
        d = pytz.UTC.localize(d)
        d = d.astimezone(pst)
        
        lat.append(float(lat_))
        long.append(float(lng_))
        hour.append(d.hour)
        day.append(d.day)
        month.append(d.month)
        weekday.append(d.weekday() + 1)
        minute.append(d.minute)
        #date.append(d.date())
        
    lat = np.array(lat).reshape((-1,1))
    long = np.array(long).reshape((-1,1))
    day = np.array(day).reshape((-1,1))
    hour = np.array(hour).reshape((-1,1))
    #date = np.array(date).reshape((-1,1))
    month = np.array(month).reshape((-1,1))
    weekday = np.array(weekday).reshape((-1,1))
    minute = np.array(minute).reshape((-1,1))
    #date = np.array(date).reshape((-1,1))
    
    
    ### [id, lat, long, weekday, month, day, hour, minute]
    latlng = np.concatenate((lat, long), axis = 1)
    
    order_time = np.concatenate((weekday, month), axis = 1)
    order_time = np.concatenate((order_time, day), axis = 1)
    order_time = np.concatenate((order_time, hour), axis = 1)
    order_time = np.concatenate((order_time, minute), axis = 1)
    #order_time = np.concatenate((order_time, date), axis = 1)
    
    
    result = np.concatenate((latlng, order_time), axis = 1)
    result = np.concatenate((np.array(order_dataset['id']).reshape((-1,1)), result), axis = 1)
    
    return np.array(result)

#order_data = data_normalize(order_dataset[0:50])
    



