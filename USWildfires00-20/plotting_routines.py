import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_hexbin_frame(df, norm, corners, ll0, gridsize, bCB, cbarlims):
    """For plotting the total annual Wildfire size in acres."""

    # set up basemap
    waterclr = '#D9D9D9'; landclr  = '#717171';
    m = Basemap(llcrnrlon=corners[0], llcrnrlat=corners[1],
        urcrnrlon=corners[2], urcrnrlat=corners[3],
        area_thresh=10000.0, resolution='l',
        projection='lcc', lat_0 = ll0[0], lon_0 = ll0[1])
    m.drawcountries()
    m.drawcoastlines(color='0.2')
    m.drawmapboundary(fill_color=waterclr)
    m.fillcontinents(color=landclr,lake_color=waterclr)

    # Put lat & long in basemap format
    x, y = m(df['LONGITUDE'], df['LATITUDE'])
    # Normalize by number of yrs. to get total fire size in 1000's per yr.
    colours = df['FIRE_SIZE']/norm

    h = m.hexbin(x, y, C=colours, gridsize=gridsize,bins='log',
            reduce_C_function = np.sum,
            cmap = 'YlOrRd', vmin=cbarlims[0], vmax=cbarlims[1])

    # frrom input: only do a colourbar on some frames
    if(bCB):
        cb = m.colorbar(location='top')
        cb.set_label(label="Total area affected by wildfire [acres/year]", size=13)

        cb.ax.tick_params(labelsize=11)
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        #cb.ax.set_yticks([0.1, 1, 10, 100, 1000])
        #cb.ax.set_yticklabels(['0.1', '1', '10', '100', '1000'])

    # return the PolyCollection output by hexbin() to be used in main notebook
    return h

def plot_hexbin_frame_diff(df, norm, corners, ll0, gridsize, bCB, cbarlims):
    """For plotting the percent change in annual Wildfire size."""

    # set up basemap
    waterclr = '#D9D9D9'; landclr  = '#717171';
    m = Basemap(llcrnrlon=corners[0], llcrnrlat=corners[1],
        urcrnrlon=corners[2], urcrnrlat=corners[3],
        area_thresh=10000.0, resolution='l',
        projection='lcc', lat_0 = ll0[0], lon_0 = ll0[1])
    m.drawcountries()
    m.drawcoastlines(color='0.2')
    m.drawmapboundary(fill_color=waterclr)
    m.fillcontinents(color=landclr,lake_color=waterclr)

    # Put lat & long in basemap format
    x, y = m(df['LONGITUDE'], df['LATITUDE'])

    # make a subset df in which the FIRE_SIZE for fires from '00-'09 are 
    # multiplied by -1, to count them as negatives in the hexbin sum
    df_copy = df[["FIRE_YEAR", "FIRE_SIZE"]]
    condition = ((df_copy['FIRE_YEAR'] >= 2000 )&(df_copy['FIRE_YEAR'] < 2010))
    df_copy.loc[ condition, 'FIRE_SIZE' ] = \
            -1.0*df_copy.loc[ condition, 'FIRE_SIZE' ]
    # Normalize by number of yrs. to get total fire size per yr.
    colours = df_copy['FIRE_SIZE']/norm

    h = m.hexbin(x, y, C=colours, gridsize=gridsize, cmap='seismic',
            reduce_C_function = np.sum,          
            vmin=cbarlims[0], vmax=cbarlims[1])

    # frrom input: only do a colourbar on some frames
    if(bCB):
        cb = m.colorbar(location='bottom')
        cb.set_label(label="Percent change in acres burned between \'10-\'19 and \'00-\'09", size=12)
        cb.ax.tick_params(labelsize=12)

    # return the PolyCollection output by hexbin() to be used in main notebook
    return h

