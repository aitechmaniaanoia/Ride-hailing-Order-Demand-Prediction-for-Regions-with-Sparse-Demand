# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:45:37 2019

@author: World peace
"""

import data_processing
import grid_index
import pandas as pd
import numpy as np

## read dataset
order_dataset = pd.read_csv('test_data.csv', encoding='ISO-8859-1')

order_data = data_processing.data_normalize(order_dataset[0:])
grid_index = grid_index.trans_index(order_data[:,1],order_data[:,2])

cnn_train = np.concatenate((order_data[:,4:], grid_index), axis = 1)
cnn_data = pd.DataFrame(data = cnn_train, columns = ['month', 'day', 'hour', 'minute', 'y_idx', 'x_idx'])

#cnn_data['minute_bin'] = cnn_data['minute'].apply(lambda x:0 if x<30 else 1)
 
###########################

#demand_groups = cnn_data.groupby(['month','day','hour','minute_bin','x_idx','y_idx'])
demand_groups = cnn_data.groupby(['month','day','hour','x_idx','y_idx'])

def get_neighborhood( x, y ):
    S = 5
    return [ (x_, y_) for y_ in range(y-S, y+S) for x_ in range(x-S,x+S) ]

num_days = [31,28,31,30]
maps = []
for month in range(1,4+1):
    for day in range(1,num_days[month-1]+1):
        print(day,'/',num_days[month-1])
        for hour in range(24):
            #for minute_bin in [0,1]:
            heatmap = [[0 for _ in range(50)] for _ in range(30)]
            for x in range(0,50):
                for y in range(0,30):
                    try:
                        heatmap[y][x] = demand_groups.get_group( (month, day, hour, x, y) ).size/6 # counts 1 per column, so 7 per order
                            ## columns are backwards?? x is y?
                            #heatmap[y][x] = demand_groups.get_group( (month, day, hour, x, y) ).size/7 # counts 1 per column, so 7 per order
                    except KeyError:
                        heatmap[y][x] = 0
            heatmap = np.array(heatmap)
            maps.append(heatmap)
maps = np.array(maps)
np.savez_compressed("test_heatmaps.npz", maps)
