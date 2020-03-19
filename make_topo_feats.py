import grid_index
import datetime
import weather
import numpy as np

#import holidays

topo_embeddings_f = open("topo_embedding")
topo_embeddings = topo_embeddings_f.read()
topo_embeddings_f.close()

topo_embeddings = topo_embeddings.split("\n")[1:-1]
topo_embeddings = np.array([
          np.array(embed.split(" ")[1:-1]).astype(np.float) for embed in topo_embeddings
        ])

#print(topo_embeddings)
maps = np.load("demand_smaller.npz")
data=maps['arr_0']
summap = data.sum(axis=0)
#print(summap.shape)
maps.close()

SEQ_LEN = 6
THRESHOLD = 1

topo_embeddings_seq = []
for y in range(data.shape[1]):#30
  for x in range(data.shape[2]):#50
    if summap[y][x]<25:
        continue
    index = data.shape[2]*y + x
    embedding = topo_embeddings[index] # x,y embedding
    topo_embeddings_seq.append(embedding)
topo_embeddings_seq = np.array(topo_embeddings_seq)
print(topo_embeddings_seq)
print(topo_embeddings_seq.shape)
# For training example n, use row topo[n%(30*50)]
np.savez_compressed("topo.npz", topo_embeddings_seq)
