import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_routines as pr
import calc_routines as cr


def analyze_conditions(fulldf, df0, collist, colvalues,
        dist_edges, angle_step,
        imgdirpref, imgdirsuffs,
        iCondPlot, condimgtitle, condimgfile,
        bPlots=[False, False, False],
        bPrints=[False, False, False]):
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

    # True/False values for determining whether certain plots are made
    bMakeDataPrints     = bPrints[0]
    bMakeCondDataPrints = bPrints[1]
    bMakeCheckPrints    = bPrints[2]

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
    h0_shots, h0_misses, h0_goals, h0_SAT, angle_edges = \
        cr.calculate_all_hists(df0, dist_edges, dist_step, angle_step,
                iPrints)

    if(bMakeCtrPlots):
        imgdir = imgdirpref + imgdirsuffs[0]

        # Call a routine that makes contour + hist maps for each
        # of the events: SAT, SOG, miss, goals
        pr.make_all_ctr_hist_plots(df0, imgdir,
            h0_SAT, h0_shots, h0_misses, h0_goals,
            dist_edges, angle_edges, h0_SAT)

    Ngoals0 = len( fulldf.loc[ fulldf['event'].isin(['GOAL'])].index)
    NSAT0 = len(fulldf.index)
    Spercent0 = Ngoals0/NSAT0

    # For now, I want NaN's in xG0. It informs where the invalid
    # histogram regions are
    with (np.errstate(divide='ignore', invalid='ignore')):
        xG0 = h0_goals/h0_SAT

    if(bMakexGPlots):
        imgstr = imgdirpref + imgdirsuffs[1] + 'xG-ref.png'
        pr.plot_single_histogram(xG0, dist_edges, angle_edges,
                h0_SAT, '', [Ngoals0, NSAT0],
                imgstr=imgstr)


    # ------- Slices of the full data Set ------- #
    # Analyze "sub-dataframes"--subdf--which are created from specific
    # slices of the full data set based on values within columns

    print('\n----------------------------------')
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
                pr.make_all_ctr_hist_plots(subdf, imgdir,
                    hs_SAT, hs_shots, hs_misses, hs_goals,
                    dist_edges, angle_edges, h0_SAT,
                    pr.titledict, col, colvals[i])

            # ----------------------- delta xG -------------------------- #

            # Calculate a few metrics to explore the effects of each
            # value on the expected goals histogram
            diff, var, deltaxG = cr.calculate_xG_diffvardelta(
                    xGs, hs_SAT, xG0)

            # Store some statistics/metrics
            differences[i]   = diff
            variances[i]     = var
            maxdeviations[i] = np.max(np.abs(deltaxG))
            numrecords1[i]   = len(subdf.index)
            numrecords2[i]   = np.sum( hs_SAT[1:hs_SAT.shape[0],:] )
            numgoals[i]      = \
                    len( subdf.loc[ subdf['event'].isin(['GOAL'])].index)
            shpcts[i]        = (numgoals[i]/numrecords1[i])

            if(bMakeDataPrints):
                # Print some statistics/metrics
                print(f'weighted diff = {diff:.4e},\tvariance = {var:.4e}')
                #print(f'max deviation = {maxdeviations[i]}')
                print(f'num shots(1) = {numrecords1[i]:d}, num shots(2) = {numrecords2[i]}, num goals = {numgoals[i]}')
                print(f'shooting percentage = {numgoals[i]/numrecords1[i]}')


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

            if(bMakeCondDataPrints):
                print(f'S%diff = {shpcts[i]-Spercent0:0.2e};\t'
                    f'n_sat weighted diff = {diff:0.2e}\n')

        if(bMakeCheckPrints):
            print('\nChecking counts for the number of shots:')
            print('Full dataset\t Reference dataset (df0)')
            print(f'{len(fulldf.index)}\t\t{len(df0.index)}')
            print('sum(len(subdf.index))\tsum(shot histogram)')
            print(f'{np.sum(numrecords1)}\t\t\t{np.sum(numrecords2)}')
            print('Note: the last two numbers can be different due to the fact that a small number of shots are beyond that largest-radius histogram bin.')

        # Store these statistics/metrics for each column in a list
        alldiffs.append(differences)
        allvars.append(variances)
        allmaxdev.append(maxdeviations)
        allngoals.append(numgoals)
        allnshots.append(numrecords1)
        allshpcts.append(shpcts)

        if(bMakeCondDataPrints):
            normShpercentDiff = np.sum(np.abs(shpcts-Spercent0))/len(shpcts)
            normWeightDiff = np.sum(np.abs(differences))/len(differences)
            print(f'sum|S% diff| = {normShpercentDiff:0.2e};\t'
                    f'sum|weighted diffs| = {normWeightDiff:0.2e}')

    print('---------------------------------')


    # Make a plot of the statistics/metrics for each column & value
    if(bMakeSctrPlots):
        imgstr = imgdirpref + condimgfile
        pr.conditions_plot(collist, allcolvals,
            alldiffs, allshpcts, Spercent0, iCondPlot,
            title=condimgtitle, imgstr=imgstr)



