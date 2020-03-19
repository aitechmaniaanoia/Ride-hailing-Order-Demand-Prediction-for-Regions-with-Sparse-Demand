import tslearn.metrics
import numpy as np

maps = np.load("demand.npz")
maps_=maps['arr_0']
maps.close()

# See page 4 of paper (Semantic view):
# want to compare average weekly demand time series for each location
# round off to nearest whole week: 17472 half-hour blocks
# reshape to 336=48*7 = one week blocks
# Transpose: we have a year's worth of samples for each half-hour block
# mean: average demand in each time block
# result is average weekly demand pattern
nodes = [np.mean( maps_[:17472,i,j].reshape( (-1,336) ).T, axis=1 ) for i in range(30) for j in range(50)]

#print(nodes)
outfile = open("graph.txt","w")
similarity = [[0 for _ in range(50*30)] for _ in range(50*30)]
for i in range(50*30):
    print(i)
    for j in range(i,50*30):
        #print(i,j/1500)
        sim = np.exp(-1 * tslearn.metrics.dtw(nodes[i], nodes[j]) )
        outfile.write( "%d %d %f\n"%(i,j,sim) )
        outfile.write( "%d %d %f\n"%(j,i,sim) )
outfile.close()

# embed this with LINE:
# https://github.com/tangjianpku/LINE
