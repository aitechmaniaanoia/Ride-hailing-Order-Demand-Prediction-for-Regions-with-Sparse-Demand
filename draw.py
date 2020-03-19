import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

maps = np.load("demand.npz")
maps_=maps['arr_0']
maps.close()

fig = plt.figure()
ax = plt.axes(xlim=(-0.5, 49.5), ylim=(-0.5, 29.5))

"""
# draw with map as background
# TODO make the map fit the bg more precisely
heatmap = plt.imshow( np.mean(maps_,axis=0), cmap='hot', interpolation='nearest', animated=True )
from PIL import Image
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
planeImg = Image.open("bg/bg.png")#'RGB', (20, 50), color = 'red')
planeImg.putalpha(64)
imagebox = OffsetImage( planeImg , zoom=0.74 )
ab = AnnotationBbox( imagebox, (25,16), xybox=( 0., 0. ), xycoords='data', boxcoords='offset points', frameon=False )
ax.add_artist(ab)
plt.colorbar(heatmap)
plt.show()
"""

#!/usr/bin/env python3
import matplotlib as mpl
from matplotlib import animation as anim
import sys
import random

avgs = np.array([np.mean(maps_[i::48],axis=0) for i in range(48)])
arr = maps_[0] #np.zeros((50, 30), dtype = int)
#arr = avgs[0]

#cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                    #['black','green','white'],
                                                    #256)
#bounds=[0,0,10,10]
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

im=plt.imshow(arr, interpolation='nearest',
        cmap = 'hot', #vmin=0, vmax=255,
              origin='lower', animated=True) # small changes here

time = 0

#def stand(arr):
    #global time
    #arr = maps_[time]
    #return arr



#time_text = ax.text(1, -1,'')#,horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
def initf():
    #time_text.set_text('start')
    return im,# time_text

def time_to_str(t):
    mon_lens = [31,28,31,30,31,30,31,31,30,31,30,31]
    #mon_lens = [2 for _ in range(12)]
    mon_lens = [2*24*l for l in mon_lens]
    i = 0
    while sum(mon_lens[:i+1]) < t:
        i+=1
    if i > 0:
        t = t - sum(mon_lens[:i])
    day = 1+(t//48)
    hour = (t%48)//2
    half = 30*(t%2)
    return "2019/%d/%d %02d:%02d"%(i+1,day,hour,half)

def animate(i):
    global time
    #arr=im.get_array()
    time+=1
    if time > 24*2:
        time = 0
        # loop in january for now
    #arr = maps_[time]
    arr = avgs[time%len(avgs)]
    #arr=5*np.mean(maps_,axis=0)
    #walk()
    ##print(a)
    #arr = stand(arr)
    im.set_array(arr)
    #time_text.set_text(str(time))
    plt.title(time_to_str(time).split(" ")[1])
    #time_text = im.axes.text(1, -1,'test')#,horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    #time_text.set_text("hi")
    return im,# time_text

anim = anim.FuncAnimation(fig, animate, frames=48, interval=  5, blit=False, init_func=initf)

#anim.save('average.gif', writer='imagemagick', fps=24)
#anim.save('jan.gif', writer='imagemagick', fps=24)
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
#anim.save('jan.mp4', writer=writer)

plt.show()
