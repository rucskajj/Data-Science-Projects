import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr

def check_diff_calc(h1g, h1a, h2g, h2a, h0g, h0a, d1, d2):
    """
    An amusing equivalence between a more complicated sum of the histogram values, and 
    the sum of two difference values (calculate_xG_diffvar)
    for two subsets which, together, represent the total data set. E.g.,
    df1=df['bParam'=True] and df2=df['bParam'=False].

    Routine is not currently used, but could be interesting later.
    """

    [M,N] = h0a.shape
    checkg1 = h1g[1:M,:]
    checka1 = h1a[1:M,:]
    checkg2 = h2g[1:M,:]
    checka2 = h2a[1:M,:]
    checkg0 = h0g[1:M,:]
    checka0 = h0a[1:M,:]

    term1 = np.sum(checkg1)/np.sum(checka1)
    term2 = np.sum(checkg2)/np.sum(checka2)

    with (np.errstate(divide='ignore', invalid='ignore')):
        term3_1 = (checkg0/checka0)
    term3_1[np.isnan(term3_1)] = 0
    term3_2 = checka1/np.sum(checka1)
    term3_3 = checka2/np.sum(checka2)
    term3 = np.sum(term3_1*(term3_2 + term3_3))

    checksum = term1 + term2 - term3
    print('Check for equivalence:',(d1+d2), checksum)



def calculate_xG_diffvardelta(xG1, a1, xG0):
    """
    A routine for quantifying how different a given xG1 is from xG0.

    xG1: expected goals histogram for some subset of the data.
    xG0: expected goals histogram for the full data set.
    """

    [M, N] = xG1.shape
    #print([M,N],xG1[1:M,:].shape)

    # Because of my choice to combine the 0th and 1st columns in the histograms
    xG1_v = xG1[1:M,:]
    a1_v  = a1 [1:M,:]
    xG0_v = xG0[1:M,:]

    # xG values will have nans
    xG1_v[np.isnan(xG1_v)] = 0
    xG0_v[np.isnan(xG0_v)] = 0

    # Difference between xG1 histogram and the full date set xG0 histogram

    # A simple difference xG1-xG0, weighted by the number of
    # events (shots, misses and goals) used to calcualte the xG1 histogram
    weighted_diff = np.sum( a1_v * (xG1_v - xG0_v)) / np.sum(a1_v)

    # weighted square of that difference
    variance = np.sqrt( np.sum( a1_v * (xG1_v - xG0_v)**2 ) / np.sum(a1_v))

    # calculate deltaxG only where shots were recorded in the histogram
    deltaxG = np.zeros(a1.shape)
    diffinds = a1 > 20 # indicies for bins with > 10 shots in them
    deltaxG[diffinds] = xG1[diffinds] - xG0[diffinds]

    # Weight by .... ?
    # number of shots: now this tracks a number with units of goals
    # num shots/mean num shots in subset
    # num shots/total shot in subset
    #deltaxG = (xG1 - xG0)*(a1)/np.mean(a1)
    #deltaxG = (xG1 - xG0)*(a1)/np.sum(a1)

    return weighted_diff, variance, deltaxG

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
