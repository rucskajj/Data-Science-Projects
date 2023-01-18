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

    # from input: only do a colourbar on some frames
    if(bCB):
        cb = m.colorbar(location='top')
        cb.set_label(label="Total area affected by wildfire [acres/year]", size=13)

        cb.ax.tick_params(labelsize=11)
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')

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

def plot_firesize_overtime(med, mean, summ, count, title):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=[4.5,6])
    plt.subplots_adjust(hspace=0.0)

    ax = axs[0]

    ax.plot(med, '.-', color='#16b51e',  lw=2)
    ax.set_ylabel("Median fire size [acres]", fontsize=12, color='#16b51e')
    ax2=ax.twinx()
    ax2.plot(mean, '.-', color='#616160', lw=2)
    ax2.set_ylabel("Mean fire size [acres]", fontsize=12, color='#616160')

    ax.grid(visible=True, zorder=-1, alpha=0.3)
    ax.set_title(title, pad=15)
    
    ax = axs[1]
    ax.plot(count, '.-', color='#e8dd0c', lw=2)
    ax.set_ylabel("Number of fires", fontsize=12, color='#e8dd0c')
    ax2=ax.twinx()
    ax2.plot(summ/1e6, '.-', color='C4', lw=2)
    ax2.set_ylabel("Sum of wildfire size [millions of acres]", fontsize=10, color='C4',labelpad=15)
    ax2.ticklabel_format(style='plain',axis='y')
    
    ax.set_xlim([med.index.min(), med.index.max()])
    ax.set_xlabel("Year", fontsize=14, labelpad=10)
    
    ax.grid(visible=True, zorder=-1, alpha=0.3)
    
    plt.show()

def plot_firesizeclass_2frame(subdf, df_lrg, labelstr, size_classes):
    fig, axs = plt.subplots(1,2, sharey=True, figsize=[10,4])
    plt.subplots_adjust(wspace=0.0)
    
    alphas = [1.0,0.5,0.35,0.2]
    colours = ["C3", "k", "k", "k"]
    zorders = [1, 0, 0, 0]
    ai = 0
    ax = axs[0]
    for sc in np.flip(size_classes):
        df1 = df_lrg.loc[(df_lrg['FIRE_SIZE_CLASS'] == sc),['FIRE_YEAR', 'FIRE_SIZE']]
        sumfsize = df1.groupby(['FIRE_YEAR'])['FIRE_SIZE'].sum()#.to_numpy()
        #print(totfiresize)
        ax.plot(sumfsize/np.mean(sumfsize), ".-", lw=2,
                color=colours[ai], alpha=alphas[ai], zorder=zorders[ai],
                label=sc)
        ai += 1
    ax.hlines(1.0, sumfsize.index.min(), sumfsize.index.max(),
              color="k", ls="--", alpha=0.4, zorder=0)
        
    xticks = ax.xaxis.get_major_ticks()
    xticks[-2].label1.set_visible(False)
    ax.text(0.70, 0.90, "All fires", fontsize=14, transform=ax.transAxes)
    ax.set_xlim([sumfsize.index.min(), sumfsize.index.max()])
    ax.legend(title=" U.S. Fire\nSize Class",frameon=False, loc='upper left')

    ai = 0
    ax = axs[1]
    for sc in np.flip(size_classes):
        df1 = subdf.loc[(subdf['FIRE_SIZE_CLASS'] == sc),['FIRE_YEAR', 'FIRE_SIZE']]
        sumfsize = df1.groupby(['FIRE_YEAR'])['FIRE_SIZE'].sum()#.to_numpy()
        #print(totfiresize)
        ax.plot(sumfsize/np.mean(sumfsize), ".-", lw=2,
                color=colours[ai], alpha=alphas[ai], zorder=zorders[ai],
                label=sc)
        ai += 1
        
    ax.hlines(1.0, sumfsize.index.min(), sumfsize.index.max(),
              color="k", ls="--", alpha=0.4, zorder=0)
        
    ax.text(0.10, 0.90, labelstr, fontsize=14, transform=ax.transAxes)
    ax.set_xlim([sumfsize.index.min(), sumfsize.index.max()])
    
    fig.add_subplot(111, frameon=False)
    plt.xticks([]); plt.yticks([]);
    plt.xlabel("Year",fontsize=14, labelpad=24)
    plt.title("Ratio of sum of wildfire size to the average from the full '92-'20 period", fontsize=12)
