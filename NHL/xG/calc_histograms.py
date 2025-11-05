import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr


def calculate_all_hists(df, dist_edges, dist_step, angl_step, iPrints):
    '''
    Calculates 2D histograms of the shot location data within the
    dataframe df. Histogram axes are shot distance and angle. Histograms
    are split by "event" type: i.e. 'SHOT', 'MISS', 'GOAL'.

    Returns 2D numpy arrays with the histogram data.
    '''


    # Some boolean values to turn on/off the plots & prints I used
    # to develop the script
    if(iPrints == 0):
        bDevPlots = False
        bDevPrints = False
    elif(iPrints == 1):
        bDevPlots = False
        bDevPrints = True
    elif(iPrints == 2):
        bDevPlots = True
        bDevPrints = False
    elif(iPrints == 3):
        bDevPlots = True
        bDevPrints = True  


    # Bins for the angle
    # Breaking angles into <= 90 and > 90.
    # <= 90 gets several bins, all of > 90 will eventually be in one bin in the
    # final histogram.
    Nstep_l90 = int(90/angl_step)+2
    Nstep_g90 = int(180/angl_step)-Nstep_l90+2
    angl_edges_l90 = np.linspace(0, 90+angl_step, Nstep_l90)
    # > 90 will have a histogram with several bins in an intermediate step, a
    # 2D hist plot is available to see what the >90 dist looks like.
    angl_edges_g90 = np.linspace(90+angl_step, 180, Nstep_g90)

    if(bDevPrints):
        print(Nstep_l90, Nstep_g90)
        print(dist_edges, angl_edges_l90, angl_edges_g90, dist_step, angl_step)


    # --------- Calculate histograms ---------------------------------------- #
    # Within each calculate_single_hist() call, every event in event list is
    # loaded into a sub-dataframe on which the histogram is calculated

    eventlist = ['SHOT']
    hist_shots = calculate_single_hist(
        df, eventlist, dist_edges, dist_step, angl_edges_l90, angl_edges_g90,
        bDevPlots, bDevPrints)
    
    eventlist = ['MISS']
    hist_misses = calculate_single_hist(
        df, eventlist, dist_edges, dist_step, angl_edges_l90, angl_edges_g90,
        bDevPlots, bDevPrints)

    eventlist = ['GOAL']
    hist_goals = calculate_single_hist(
        df, eventlist, dist_edges, dist_step, angl_edges_l90, angl_edges_g90,
        bDevPlots, bDevPrints)

    eventlist = ['SHOT', 'MISS', 'GOAL']
    hist_all = calculate_single_hist(
        df, eventlist, dist_edges, dist_step, angl_edges_l90, angl_edges_g90,
        bDevPlots, bDevPrints)

    return hist_shots, hist_misses, hist_goals, hist_all, angl_edges_l90

def calculate_single_hist(df, eventlist,
    dist_edges, dist_step, angl_edges_l90, angl_edges_g90,
    bDevPlots, bDevPrints):
    '''
    Calculates one 2D histogram of the shot location data within the
    dataframe df, according to the events within event list.
    '''


    angl_step = angl_edges_l90[1] - angl_edges_l90[0]
    Nstep_l90 = len(angl_edges_l90)

    # "sub" dataframe for <= 90 shots
    subdf_l90 = df.loc[(df['event'].isin(eventlist)) &
            (np.abs(df['angle'])<90+angl_step) ]

    # 2d hist for <= 90 shots
    [hist_l90, xedges, yedges] = np.histogram2d(
            subdf_l90['distance'], np.abs(subdf_l90['angle']),
            bins=(dist_edges,angl_edges_l90))

    # sub df for > 90
    subdf_g90 = df.loc[(df['event'].isin(eventlist)) &
            (np.abs(df['angle'])>=90+angl_step) ]

    # hist for > 90
    [hist_g90, xedges, yedges] = np.histogram2d(
            subdf_g90['distance'], np.abs(subdf_g90['angle']),
            bins=(dist_edges,angl_edges_g90))

    if(bDevPlots):
        # plot the 2D hist for > 90 shots, if you're curious
        fig, ax = plt.subplots(1,1, facecolor='w', edgecolor='k')
        extents = (dist_edges.min(), dist_edges.max(),
            angl_edges_g90.min(), angl_edges_g90.max())
        imgplot = ax.imshow(hist_g90.T, extent=extents, origin='lower')
        fig.colorbar(imgplot)
        #plt.show()
        plt.close()


    #print(np.sum(hist_g90,axis=1).shape)
    # [:,Nstep_l90-2] grabs the last row (90-95) in the <= 90 histogram 
    # sum(hist_g90,...) adds the (95-180) data to this row
    hist_l90[:,Nstep_l90-2] += np.sum(hist_g90,axis=1) 

    sumcheck1 = int(np.sum(hist_l90)) # to check this calculation later

    # hist_l90[0,:] is the first column of data, in the plot (smallest r bin)
    col0data = np.copy(hist_l90[0,:])
    hist_l90[0,:] += hist_l90[1,:]
    hist_l90[1,:] += col0data
    # I am effectively givin the first two columns (first two r_bins) the same
    # values, for aesthetic purposes in the final histogram. There is not a
    # lot of data in the first distance bin, but there is some.

    [M,N] = hist_l90.shape
    # sum over only part of the hist, since the first two columns duplicate data
    sumcheck2 = int(np.sum(hist_l90[1:M,:])) 
    if(bDevPrints):
        print(f'Total counts in the histogram, checked two times:\
                {sumcheck1}, {sumcheck2}')
        print(f'Total number of entries in the original dataframe:\
                {len(subdf_l90.index)+len(subdf_g90.index)}')

    if(bDevPlots):
        fig, ax = plt.subplots(1,1, facecolor='w', edgecolor='k')
        extents = (dist_edges.min(), dist_edges.max(),
                angl_edges_l90.min(), angl_edges_l90.max())
        imgplot = ax.imshow(hist_l90.T, extent=extents, origin='lower')
        fig.colorbar(imgplot)
        plt.show()
        #plt.close()
    
    return hist_l90
