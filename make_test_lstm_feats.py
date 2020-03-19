import grid_index
import datetime
import weather
import numpy as np

import holidays

maps = np.load("demand_smaller.npz")
training_data=maps['arr_0']
maps.close()

do_2019 = True
if not do_2019:
  #maps = np.load("demand.npz")
  data = training_data
  start_time = datetime.datetime(2018, 1, 1)
else:
  maps = np.load("test_heatmaps.npz")
  data=maps['arr_0']
  maps.close()
  start_time = datetime.datetime(2019, 1, 1)

ca_holidays = holidays.Canada(prov='BC')

SEQ_LEN = 6

if do_2019:
  timename = "lstm_time_2019.npz"
  weathername = "lstm_weather_2019.npz"
else:
  timename = "lstm_time.npz"
  weathername = "lstm_weather.npz"

precompute = False
if precompute:
  times_of_interest = []
  time_features = []
  weather_features = np.zeros( (data.shape[0], 30, 50, 5) )
  for time in range(data.shape[0]):
      # log progress
      if time%100 == 0:
          print(100*time / data.shape[0])
  
      time_obj = (start_time + time*datetime.timedelta(hours=0.5))
      time_str = time_obj.isoformat("-")[:10]
  
      new_feats = np.array([
          time_obj.weekday(),
          time_obj.month,
          time_obj.hour,
          int(time_str in ca_holidays),
          #time_obj.minute // 30,
      ])
      time_features.append( new_feats ) # = np.concatenate( [time_features, new_feats], axis=0 )
  
      # Precompute all weather features
      for x in range(50):
          for y in range(30):
              weather_features[time,y,x] = weather.get_weather( 
                  (x,y),
                  time_str
              )
  
  time_features = np.array(time_features)
  print(time_features.shape)
  
  np.savez_compressed(timename, time_features)
  np.savez_compressed(weathername, weather_features)
  exit()
else:
  feats = np.load(timename)
  time_features=feats['arr_0']
  feats.close()
  feats=np.load(weathername)
  weather_features=feats['arr_0']
  feats.close()
  
  N_CHUNKS = 1
  total_len=data.shape[0]-SEQ_LEN
  chunk_size = 1+(total_len//N_CHUNKS)
  print(data.shape)
  sum_data = np.sum(training_data, axis=0)
  print(sum_data.shape)
  for chunk in range(N_CHUNKS):
      print()
      print(chunk)
      lstm_train_data = []
      start_time = int(chunk*chunk_size)
      end_time = start_time + chunk_size
      if end_time > total_len:
          end_time = total_len
      print(end_time)
      for time in range(start_time, end_time):
          if time % 100 == 0:
              print(time*100 / data.shape[0])
          for y in range(data.shape[1]):
              for x in range(data.shape[2]):
  
                  if sum_data[y,x] < 25:
                      continue
  
                  pos_feats = np.array([x,y])

                  sequence = np.array([
                          np.concatenate(
                              [
                                  weather_features[t,y,x],
                                  pos_feats,
                                  time_features[t]
                              ], axis=0
                          )
                          for t in range(time, time + SEQ_LEN)
                      ])
                  lstm_train_data.append(sequence)
      lstm_train_data = np.array(lstm_train_data)
              
      fname = ("lstm_%d"%(chunk)) + ("_test" if do_2019 else "") + ".npz"
      np.savez_compressed(fname, lstm_train_data)
