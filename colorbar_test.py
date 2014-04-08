# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:41:08 2014

@author: sb238920
"""

import numpy as np
#import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#data = np.random.random((10,10))
#data = 10 * (data - 0.8)

#fig, ax = plt.subplots()
#norm = MidpointNormalize(midpoint=0)
#im = ax.imshow(data, norm=norm, cmap=plt.cm.seismic, interpolation='none')
#fig.colorbar(im)
#plt.show()