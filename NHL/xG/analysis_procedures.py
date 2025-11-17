import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr
import calc_routines as cr


def analyze_conditions(fulldf, df0, collist, colvalues,
        dist_edges, angle_step,
        imgdirpref, imgdirsuffs, iCondPlot, condimgstr,
        bPlots=[False, False, False], bPrints=False):
    '''
    A routine which makes a number of plots comparing slices of some
    larger dataframe (fulldf) to a reference dataframe (df0).

    The dataframes are generated from .csv's of MoneyPuck's NHL shot data.
    '''

    # --------- Bookkeeping ------------ #

    # True/False values for determining whether certain plots are made
    bMakeCtrPlots  = bPlots[0]
    bMakexGPlots   = bPlots[1]
    bMakeSctrPlots = bPlots[2]

    # Turns on/off prints & plots for debugging purposes within the
    # histogram routines
    iPrints = 0

    # Grab the step in the distance bins
    dist_step = dist_edges[1]-dist_edges[0]

    # ----------------- Calculating histograms ------------------------- #


    # ------- Reference, "0", Data Set ------- #
    # The delta xG histogram and associated derived values come from
    # comparing "dataframes" to this reference data frame
    # df0 is passed into this routine

    # Calculate the histograms for every event in the data set
    h0_shots, h0_miss, h0_goal, h0_SAT, angle_edges = \
        cr.calculate_all_hists(df0, dist_edges, dist_step, angle_step,
                iPrints)

    # For now, I want NaN's in xG0. It informs where the invalid
    # histogram regions are
    with (np.errstate(divide='ignore', invalid='ignore')):
        xG0 = h0_goal/h0_SAT


    # ------- Slices of the full data Set ------- #
    # Analyze "sub-dataframes"--subdf--which are created from specific
    # slices of the full data set based on values within columns

    print('\n-------------------------')
    print('Analyzing the following subsets/slices:')


    # Empty lists to append summary data into
    alldiffs   = []
    allvars    = []
    allmaxdev  = []
    allnshots  = []
    allngoals  = []
    allcolvals = []
    allshpcts  = []

    for col in collist:
        # Desired values for each column are passed in via colvalues
        colvals = colvalues[col] 

        allcolvals.append(list(colvals))

        Ncolvals = len(colvals)

        # Empty arrays to store summary data for this column
        differences   = np.zeros(Ncolvals)
        variances     = np.zeros(Ncolvals)
        maxdeviations = np.zeros(Ncolvals)
        numgoals      = np.zeros(Ncolvals, dtype=int)
        numrecords1   = np.zeros(Ncolvals, dtype=int)
        numrecords2   = np.zeros(Ncolvals, dtype=int)
        shpcts        = np.zeros(Ncolvals)

        print('\n----------')
        
        for i in range(Ncolvals):
            print(f'{col}: {colvals[i]}')

            # Create a subdata frame based on this column & value
            subdf = fulldf.loc[fulldf[col]==colvals[i]]
            # Calculate 2D histograms for this data frame
            hs_shots, hs_misses, hs_goals, hs_SAT, angle_edges = \
                cr.calculate_all_hists(subdf, dist_edges, dist_step,
                        angle_step, iPrints)

            with (np.errstate(divide='ignore', invalid='ignore')):
                xGs = hs_goals/hs_SAT # 2D expected goals histogram

            # --------------------- Histograms and heatmaps ------------- #

            imgdir = imgdirpref + imgdirsuffs[0]
            if(bMakeCtrPlots):
                # Call a routine that makes contour + hist maps for each
                # of the events: SAT, SOG, miss, goals
                pr.make_all_ctr_hist_plots(subdf, pr.titledict, imgdir,
                    col, colvals[i],
                    hs_SAT, hs_shots, hs_misses, hs_goals,
                    dist_edges, angle_edges, h0_SAT)

            # ----------------------- delta xG ---------------------------- #

            # Calculate a few metrics to explore the effects of each
            # value on the expected goals histogram
            diff, var, deltaxG = cr.calculate_xG_diffvardelta(
                    xGs, hs_SAT, xG0)
            # deltaxG = xGs - xG0


            # Store some statistics/metrics
            differences[i]   = diff
            variances[i]     = var
            maxdeviations[i] = np.max(np.abs(deltaxG))
            numrecords1[i]   = len(subdf.index)
            numrecords2[i]   = np.sum( hs_SAT[1:hs_SAT.shape[0],:] )
            numgoals[i]      = \
                    len( subdf.loc[ subdf['event'].isin(['GOAL'])].index)
            shpcts[i]        = (numgoals[i]/numrecords1[i])

            if(bPrints):
                # Print some statistics/metrics
                print(f'diff = {diff:.4e},\tvariance = {var:.4e}')
                print(f'max deviation = {maxdeviations[i]}')
                print(f'num shots(1) = {numrecords1[i]:d}, num shots(2) = {numrecords2[i]} num goals = {numgoals[i]}')

            # Make a plot of the deltaxG histogram
            if(bMakexGPlots):
                plttitle = r"xG and $\Delta$xG for " +\
                        pr.titledict[col+'-'+str(colvals[i])]
                imgdir = imgdirpref + imgdirsuffs[1]
                imgstr = imgdir + col + '-' + str(colvals[i]) + '.png'

                # Data to be added to plot via ax.annotate()
                ann_nums = [diff, var, numgoals[i], numrecords1[i]]

                pr.plot_xG_deltaxG(xGs, deltaxG,
                        dist_edges, angle_edges, h0_SAT, plttitle,
                        imgstr=imgstr, ann_nums=ann_nums)

        if(bPrints):
            print('\nChecking counts for the number of shots:')
            print('Full dataset\tsum(len(subdf.index))\tsum(shot histogram)')
            print(f'{len(fulldf.index)}\t\t{np.sum(numrecords1)}\t\t\t{np.sum(numrecords2)}')

        # Store these statistics/metrics for each column in a list
        alldiffs.append(differences)
        allvars.append(variances)
        allmaxdev.append(maxdeviations)
        allngoals.append(numgoals)
        allnshots.append(numrecords1)
        allshpcts.append(shpcts)

    print('---------------------------------')

    Nrecords0 = len(fulldf.index)
    Ngoal0 = len( fulldf.loc[ fulldf['event'].isin(['GOAL'])].index)
    Spercent0 = Ngoal0/Nrecords0

    # Make a plot of the statistics/metrics for each column & value
    if(bMakeSctrPlots):
        imgstr = imgdirpref + condimgstr
        pr.conditions_plot(collist, allcolvals,
            alldiffs, allshpcts, Spercent0, iCondPlot, imgstr=imgstr)



