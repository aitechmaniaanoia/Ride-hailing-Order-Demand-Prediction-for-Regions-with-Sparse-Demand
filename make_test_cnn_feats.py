import grid_index
import datetime
import weather
import numpy as np

import holidays

maps = np.load("demand_smaller.npz")
training_data=maps['arr_0']
maps.close()

heatmaps_f = np.load("test_heatmaps.npz")
data = heatmaps_f['arr_0']
heatmaps_f.close()


SEQ_LEN = 6

total_len=data.shape[0]-SEQ_LEN
chunk_size = total_len
print(data.shape)
sum_data = np.sum(training_data, axis=0)
KERNEL_SIZE = 3
PAD = KERNEL_SIZE // 2
print(data.shape)
data_wide = np.pad( data, ((0,0),(PAD,PAD),(PAD,PAD)) )
shape = (KERNEL_SIZE, KERNEL_SIZE, 1)
print(sum_data.shape)
for chunk in range(1):
    print()
    print(chunk)
    cnn_train_data = []
    output_labels  = []
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

                sequence = np.array([
                    np.reshape( data_wide[t,y:y+KERNEL_SIZE,x:x+KERNEL_SIZE], shape )
                        for t in range(time, time + SEQ_LEN)
                    ])
                cnn_train_data.append(sequence)
                output_labels.append( data[time+SEQ_LEN,y,x] )
    cnn_train_data = np.array(cnn_train_data)
    output_labels = np.reshape(np.array(output_labels),(-1, 1))
            
    fname = ("Ximage_test.npz")
    np.savez_compressed(fname, cnn_train_data)
    fname = ("Ylabel_test.npz")
    np.savez_compressed(fname, output_labels)
