# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:43:50 2019

@author: World peace
"""

import numpy as np

min_lat=49.304231
max_lat=49.102290
max_lon=-122.738056
min_lon=-123.271087

interval_lat = -(max_lat-min_lat)/30
interval_lon = (max_lon-min_lon)/50

def trans_index(l_lat,l_lon):
    lidx_lat,lidx_lon = [],[]
    for i in range(len(l_lat)):
        p_lat = l_lat[i]
        p_lon = l_lon[i]
        i = 30 - abs(int((p_lat-min_lat)/interval_lat))#y
        j = int((p_lon-min_lon)/interval_lon)#x
        lidx_lat.append(i)
        lidx_lon.append(j)
        
    lidx_lat = np.array(lidx_lat).reshape((-1,1))
    lidx_lon = np.array(lidx_lon).reshape((-1,1))
    
    grid_index = np.concatenate((lidx_lat, lidx_lon), axis = 1)
    
    return grid_index

def xy_to_latlon( x, y ):
    lat = min_lat + (interval_lat * y)
    lon = min_lon + (interval_lon * x)
    return (lat, lon)
#a = trans_index(order_data[:,1],order_data[:,2])
#print(a)
