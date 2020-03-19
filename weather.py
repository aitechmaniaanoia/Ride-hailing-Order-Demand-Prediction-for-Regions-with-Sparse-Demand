import os
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from collections import defaultdict
import grid_index

WEATHER_DIR = "./data/weather/"
STATION_LOCATIONS = {
        "vancouver": (49.1825,-123.1872),
        "burnaby":   (49.2833,-122.8833),
        "richmond":  (49.1830,-123.1682),
        }
CITIES = list(STATION_LOCATIONS.keys())
WEATHER_FEATS = [
    #"Max Temp (°C)",
    #"Min Temp (°C)",
    "Mean Temp (°C)",
    "Total Rain (mm)",
    "Total Snow (cm)",
    "Total Precip (mm)",
    "Snow on Grnd (cm)",
    "Date/Time"
    ]
weather = defaultdict(list)

# Store weather data in dictionary of pandas
# dataframes. Use pandas for speed vs. finding
# items in a dictionary.
do_2019 = True
for filename in os.listdir( WEATHER_DIR ):
    if do_2019 and "2019" not in filename:
        continue
    elif not do_2019 and "2019" in filename:
        continue

    # Get city name
    city = filename[ : filename.index('.') ]
    if '_' in city:
      city = filename[ : filename.index('_') ]
    # and add city to dictionary
    weather[ city ] = defaultdict(list)

    # Load weather data
    file_path = WEATHER_DIR + filename
    data = pd.read_csv( 
            file_path, 
            sep=',', 
            quotechar='"', 
            header=[0] 
            )
    data = data[WEATHER_FEATS]
    data.set_index("Date/Time")
    data["Date/Time"] = pd.to_datetime(data["Date/Time"])
    #cols = data.columns
    #cols = [col+("_"+city if col!="Date/Time" else "") for col in cols]
    #data.columns = cols
    # Save weather data in dictionary
    weather[city] = np.array(data[WEATHER_FEATS[:-1]])

#weather_all = pd.DataFrame()
#for i,key in enumerate(weather.keys()):
    #weather_all = pd.concat( [weather_all, weather[key].iloc[:,0:(-1 if i>0 else None)]], axis=1 )
#weather_all.set_index(["Date/Time"], inplace=True)
#weather_all.fillna(0,inplace=True)
#weather_all.to_csv("weather_all.csv")

import datetime

dist_memo = dict()
def get_weather( coords, date ):
    """
    coords:  (lat, lon) tuple
    date:    date string (format "YYYY-MM-DD")
    returns: numpy array containing vector of weather
             features
    """
    if coords not in dist_memo:
        x,y=coords
        lat, lon = grid_index.xy_to_latlon( x, y )
        dists = [ 
            geodesic(coords, STATION_LOCATIONS[city]) 
            for city in CITIES 
            ]
        city = CITIES[ np.argmin( dists ) ]
        dist_memo[coords] = city
    city = dist_memo[coords]
    city_data = weather[city]
    day_no = datetime.datetime.strptime(date,"%Y-%M-%d").timetuple().tm_yday
    features = city_data[day_no]
    return features
