import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr
import calc_routines as cr


def analyze_conditions(df, collist,
        imgdirpref,
        bPlots=[False, False, False], bPrints=False):

    bMakeCtrPlots  = bPlots[0]
    bMakexGPlots   = bPlots[1]
    bMakeSctrPlots = bPlots[2]


    # ----------------- Calculating histograms ------------------------- #

    # Bins for the distance
    #dist_edges = np.linspace(0, 78, 27)
    dist_edges = np.linspace(0, 76, 20)
    dist_step = dist_edges[1]-dist_edges[0];
    angle_step = 10
    iPrints = 0


    # ------- Full Data Set ------- #

    # Calculate the histograms for every event in the data set
    h0_shots, h0_miss, h0_goal, h0_SAT, angle_edges = \
        cr.calculate_all_hists(df, dist_edges, dist_step, angle_step, iPrints)

    # For now, I want NaN's in xG0. It informs where the invalid
    # histogram regions are
    with (np.errstate(divide='ignore', invalid='ignore')):
        xG0 = h0_goal/h0_SAT


    # ------- Slices of the Data Set ------- #

    print('\n---------------------------------------------------------')
    print('Analysis for various subsets of the full data set:')


    # Empty lists to append summary data into
    alldiffs   = []
    allvars    = []
    allmaxdev  = []
    allnshots  = []
    allngoals  = []
    allcolvals = []
    allshpcts  = []

    for col in collist:
        colvals = df[col].unique() # Find all unique values for each column
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
            subdf = df.loc[df[col]==colvals[i]]
            # Calculate 2D histograms for this data frame
            hs_shots, hs_misses, hs_goals, hs_SAT, angle_edges = \
                cr.calculate_all_hists(subdf, dist_edges, dist_step,
                        angle_step, iPrints)

            with (np.errstate(divide='ignore', invalid='ignore')):
                xGs = hs_goals/hs_SAT # 2D expected goals histogram

            # --------------------- Histograms and heatmaps ------------- #

            imgdir = imgdirpref + 'Contours-Hists/'
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
                print(f'num shots = {len(subdf.index):d}, num goals = {numgoals[i]}')

            # Make a plot of the deltaxG histogram
            if(bMakexGPlots):
                plttitle = r"xG and $\Delta$xG for " +\
                        pr.titledict[col+'-'+str(colvals[i])]
                imgdir = imgdirpref + '2Dhists/'
                imgstr = None#imgdir + col + '-' + str(colvals[i]) + '.png'

                # Data to be added to plot via ax.annotate()
                ann_nums = [diff, var, numgoals[i], numrecords1[i]]

                pr.plot_xG_deltaxG(xGs, deltaxG,
                        dist_edges, angle_edges, h0_SAT, plttitle,
                        imgstr=imgstr, ann_nums=ann_nums)

        if(bPrints):
            print('\nChecking counts for the number of shots:')
            print('Full dataset\tsum(len(subdf.index))\tsum(shot histogram)')
            print(f'{len(df.index)}\t\t{np.sum(numrecords1)}\t\t\t{np.sum(numrecords2)}')

        # Store these statistics/metrics for each column in a list
        alldiffs.append(differences)
        allvars.append(variances)
        allmaxdev.append(maxdeviations)
        allngoals.append(numgoals)
        allnshots.append(numrecords1)
        allshpcts.append(shpcts)

    print('---------------------------------')

    Nrecords0 = len(df.index)
    Ngoal0 = len( df.loc[ df['event'].isin(['GOAL'])].index)
    Spercent0 = Ngoal0/Nrecords0

    # Make a plot of the statistics/metrics for each column & value
    if(bMakeSctrPlots):
        pr.conditions_plot(collist, allcolvals,
            alldiffs, allshpcts, Spercent0)

